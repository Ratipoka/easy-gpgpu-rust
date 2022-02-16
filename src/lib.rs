//! # Easy GPGPU
//! A high level, easy to use gpgpu crate based on [`wgpu`](https://github.com/gfx-rs/wgpu).
//! It is made for very large computations on powerful gpus
//! 
//! ## Main goals :
//! 
//! - make general purpose computing very simple
//! - make it as easy as possible to write wgsl shaders
//! - deal with binding buffers automatically
//! 
//! ## Limitations :
//! 
//! - only types available for buffers : bool, i32, u32, f32
//! - max buffer byte_size : around 134_000_000 (~33 million i32)
//! 
//!   use device.apply_on_vector to be able to go up to one billion bytes (260 million i32s)
//! - takes time to initiate the device for the first time (due to wgpu backends)
//! 
//! ## Example 
//! 
//! recreating [`wgpu's hello-compute`](https://github.com/gfx-rs/wgpu/tree/v0.12/wgpu/examples/hello-compute) (205 sloc when writen with wgpu)
//! 
//! ```
//! use easy_gpgpu::*;
//! fn wgpu_hello_compute() {
//!     let mut device = Device::new();
//!     let v = vec![1u32, 4, 3, 295];
//!     let result = device.apply_on_vector(v.clone(), r"
//!     fn collatz_iterations(n_base: u32) -> u32{
//!         var n: u32 = n_base;
//!         var i: u32 = 0u;
//!         loop {
//!             if (n <= 1u) {
//!                 break;
//!             }
//!             if (n % 2u == 0u) {
//!                 n = n / 2u;
//!             }
//!             else {
//!                 // Overflow? (i.e. 3*n + 1 > 0xffffffffu?)
//!                 if (n >= 1431655765u) {   // 0x55555555u
//!                     return 4294967295u;   // 0xffffffffu
//!                 }
//!                 n = 3u * n + 1u;
//!             }
//!             i = i + 1u;
//!     }
//!     return i;
//!     }
//!     collatz_iterations(element)
//!     ");
//!     assert_eq!(result, vec![0, 2, 7, 55]);
//! }
//! ```
//! => No binding, no annoying global_id, no need to use a low level api.
//! 
//! You just need to write the minimum amount of wgsl shader code.
//! 
//! ## Usage with .apply_on_vector
//! ```
//! use easy_gpgpu::*;
//! // create a device
//! let mut device = Device::new();
//! // create the vector we want to apply a computation on
//! let v1 = vec![1.0f32, 2.0, 3.0];
//! // the next line reads : for every element in v1, perform : element = element * 2.0
//! let v1 = device.apply_on_vector(v1, "element * 2.0");
//! println!("{v1:?}");
//! ```
//! ## Usage with .execute_shader_code
//! ```
//! //First create a device :
//! use easy_gpgpu::*;
//! let mut device = Device::new();
//! // Then create some buffers, specify if you want to get their content after the execution :
//! let v1 = vec![1i32, 2, 3, 4, 5, 6];
//! // from a vector
//! device.create_buffer_from("v1", &v1, BufferUsage::ReadOnly, false);
//! // creates an empty buffer
//! device.create_buffer("output", BufferType::I32, v1.len(), BufferUsage::WriteOnly, true);
//! // Finaly, execute a shader :
//! let result = device.execute_shader_code(Dispatch::Linear(v1.len()), r"
//! fn main() {
//!     output[index] = v1[index] * 2;
//! }").into_iter().next().unwrap().unwrap_i32();
//! assert_eq!(result, vec![2i32, 4, 6, 8, 10, 12])
//! ```
//! The buffers are available in the shader with the name provided when created with the device.
//! 
//! The `index` variable is provided thanks to the use of `Dispatch::Linear` (index is a u32).
//! 
//! We had only specified one buffer with `is_output: true` so we get only one vector as an output.
//! 
//! We just need to unwrap the data as a vector of i32s with `.unwrap_i32()`

use std::{vec, collections::{HashMap, HashSet}, ops};
use wgpu::{util::DeviceExt, BufferUsages};
use std::vec::Vec;
use pollster::FutureExt;

/// A buffer representation build on top of the `wgpu::Buffer`.
/// It stores a buffer name to enable a direct access in the shader.
/// To get a buffer, call `device.create_buffer` or `device.create_buffer_from`.
struct Buffer {
    buffer: Option<wgpu::Buffer>,
    name: String,
    size: u64,
    type_: BufferType,
    type_stride: i32,
    data: Option<Vec<u8>>,
    usage: BufferUsage,
    wgpu_usage: wgpu::BufferUsages,
    is_output: bool,
    output_buf: Option<wgpu::Buffer>,
}

// the maximum number of i32s we can fit in a buffer
static MAX_BYTE_BUFFER_SIZE: usize = 134_217_728;
static IDENT_SEPARATOR_CHARS: &str = " \t\n\r([,+-/*=%&|^~!<>{}";

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum BufferType {
    I32,
    U32,
    F32,
    Bool
}
impl BufferType {
    pub fn to_string(&self) -> String {
        match &self {
            BufferType::I32 => String::from("i32"),
            BufferType::U32 => String::from("u32"),
            BufferType::F32 => String::from("f32"),
            BufferType::Bool => String::from("bool")
        }
    }
    pub fn stride(&self) -> i32 {
        match &self {
            BufferType::Bool => 1,
            _ => 4
        }
    }
}


/// Defines the usage of the buffer in the wgsl shader
#[derive(Clone)]
pub enum BufferUsage {
    ReadOnly,
    WriteOnly,
    ReadWrite
}
impl ops::BitOr<BufferUsages> for BufferUsage {
    type Output = wgpu::BufferUsages;
    fn bitor(self, rhs: wgpu::BufferUsages) -> Self::Output {
        buffer_usage(self, false) | rhs
    }
}


/// Converts a high level BufferUsage to the lower level wgpu::BufferUsages
#[inline]
fn buffer_usage(usage: BufferUsage, output: bool) -> wgpu::BufferUsages {
    let out;
    match usage {
        BufferUsage::ReadOnly => {
            out = wgpu::BufferUsages::MAP_WRITE
            | wgpu::BufferUsages::STORAGE
        },
        BufferUsage::WriteOnly => {
            out = wgpu::BufferUsages::STORAGE
        },
        BufferUsage::ReadWrite => {
            out = wgpu::BufferUsages::STORAGE
        }
    }
    if output {
        out | wgpu::BufferUsages::COPY_SRC
    }else {
        out
    }
}

/// Defines a custom global_id manager, 
/// if Linear(n) is used, the index variable is automatically going to be available in the shader
/// it will range from 0 to n excluded (max n : 2^32)
#[derive(Clone)]
pub enum Dispatch {
    Linear(usize),
    Custom(u32, u32, u32)
}

/// Trait to obtain raw data of vectors in order to convert them to buffer as seamlessly as possible
/// and to obtain back a vector from the raw Vec<u8>
pub trait ToVecU8<T> {
    // returns raw content, type, type_stride, length of vec
    fn convert(v: &Vec<T>) -> (Vec<u8>, BufferType, i32, usize);
    fn get_output(v: OutputVec) -> Vec<T>;
}
impl ToVecU8<i32> for i32 {
    fn convert(v: &Vec<i32>) -> (Vec<u8>, BufferType, i32, usize) {
        (bytemuck::cast_slice::<i32, u8>(&v).to_vec(), BufferType::I32, 4, v.len())
    }
    fn get_output(v: OutputVec) -> Vec<i32> {
        v.unwrap_i32()
    }
}
impl ToVecU8<u32> for u32 {
    fn convert(v: &Vec<u32>) -> (Vec<u8>, BufferType, i32, usize) {
        (bytemuck::cast_slice::<u32, u8>(&v).to_vec(), BufferType::U32, 4, v.len())
    }
    fn get_output(v: OutputVec) -> Vec<u32> {
        v.unwrap_u32()
    }
}
impl ToVecU8<f32> for f32 {
    fn convert(v: &Vec<f32>) -> (Vec<u8>, BufferType, i32, usize) {
        (bytemuck::cast_slice::<f32, u8>(&v).to_vec(), BufferType::F32, 4, v.len())
    }
    fn get_output(v: OutputVec) -> Vec<f32> {
        v.unwrap_f32()
    }
}
impl ToVecU8<bool> for bool {
    fn convert(v: &Vec<bool>) -> (Vec<u8>, BufferType, i32, usize) {
        (v.iter().map(|&e| e as u8).collect::<Vec<_>>(), BufferType::Bool, 1, v.len())
    }
    fn get_output(v: OutputVec) -> Vec<bool> {
        v.unwrap_bool()
    }
}

/// An enum to represent the different output vectors possible
/// To get the vector back, call .unwrap_i32() for example.
#[derive(Debug)]
pub enum OutputVec {
    VecI32(Vec<i32>),
    VecU32(Vec<u32>),
    VecF32(Vec<f32>),
    VecBool(Vec<bool>)
}
impl OutputVec {
    pub fn unwrap_i32(self) -> Vec<i32> {
        match self {
            OutputVec::VecI32(val) => {
                val
            }
            _ => {
                panic!("value is not a u32!");
            }
        }
    }
    pub fn unwrap_u32(self) -> Vec<u32> {
        match self {
            OutputVec::VecU32(val) => {
                val
            }
            _ => {
                panic!("value is not a u32!, it's a {self:?}");
            }
        }
    }
    pub fn unwrap_f32(self) -> Vec<f32> {
        match self {
            OutputVec::VecF32(val) => {
                val
            }
            _ => {
                panic!("value is not a f32!");
            }
        }
    }
    pub fn unwrap_bool(self) -> Vec<bool> {
        match self {
            OutputVec::VecBool(val) => {
                val
            }
            _ => {
                panic!("value is not a bool!");
            }
        }
    }
}

pub struct ShaderModule {
    shader: wgpu::ShaderModule,
    used_buffers_id: Vec<usize>,
    // bind_group: Vec<wgpu::BindGroupEntry<'a>>,
    dispatch: Dispatch
}

pub fn remove_comments(code: String) -> String{
    let mut out = String::new();
    let mut chars = code.chars();
    while let Some(c) = chars.next() {
        if c == '/' {
            out.push(c);
            let c = chars.next();
            if Some('/') == c { // line comment
                out.pop();
                while let Some(c) = chars.next() {
                    if c == '\n' {
                        out.push(c);
                        break;
                    }
                }
            }else if Some('*') == c { // block comment
                out.pop();
                while let Some(c) = chars.next() {
                    if c == '*' {
                        if let Some('/') = chars.next() {
                            break;
                        }
                    }
                }
            }else {
                if let Some(c) = c {
                    out.push(c);
                }
            }
        }else {
            out.push(c);
        }
    }
    out
}

pub enum Command<'a> {
    Shader(ShaderModule),
    Copy(&'a str, &'a str),
    Retrieve(&'a str)
}

/// The main struct which provides abstraction over the wgpu library.
pub struct Device {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: Vec<Buffer>,
    buf_name_to_id: HashMap<String, usize>,
    output_buffers_id: Vec<usize>,
}
impl  Device {
    /// Creates a new Device. (request a device with `power_preference: wgpu::PowerPreference::HighPerformance`)
    pub fn new() -> Device {
        async {
            let instance = wgpu::Instance::new(wgpu::Backends::all());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .unwrap();
            let (device, queue) = adapter
                .request_device(&Default::default(), None)
                .await
                .unwrap();
            Device {
                adapter,
                device,
                queue,
                buffers: vec![],
                buf_name_to_id: HashMap::new(),
                output_buffers_id: vec![],
            }
        }.block_on()
    }

    /// Get some info about the gpu chosen to run the code on.
    #[inline]
    pub fn get_info(self) -> String {
        format!("{:?}", self.adapter.get_info())
    }

    /// Creates a buffer.
    /// 
    /// Arguments :
    /// 
    /// - name : the name of the buffer that will be available in the shader
    /// - data_types : the type of the buffer, can be eaiser : "i32", "u32", "f32", or "bool"
    /// - size : the element size of the buffer (if it is a buffer of 3 u32s, the size is 3)
    /// - usage : a description of how the buffer will be used in the shader
    /// - is_output : whether or not to retrieve the data in the buffer after execution
    pub fn create_buffer(&mut self, name: &str, data_type: BufferType, size: usize, usage: BufferUsage, is_output: bool) {
        let data_type_stride = data_type.stride();
        let byte_size = (data_type_stride as usize) * size;
        let id = self.buffers.len();
        // saves the position of the buffer if is_output: true, 
        // that is in order to iterates through output buffers
        if is_output {
            self.output_buffers_id.push(self.buffers.len())
        }
        // to be able to find the buffer from the name
        self.buf_name_to_id.insert(name.to_string(), id);
        // create a Buffer and adds it to the buffer vector
        self.buffers.push(Buffer {
            buffer: None,
            name: name.to_owned(),
            size: byte_size as u64,
            type_: data_type,
            type_stride: data_type_stride as i32,
            data: None,
            usage: usage.clone(),
            wgpu_usage: buffer_usage(usage, is_output),
            // if we need to get the output off the buffer, we add another one to copy to it
            is_output,
            output_buf: None
        });
    }

    /// Creates a buffer from a vector.
    /// 
    /// Arguments :
    /// 
    /// - name : the name of the buffer that will be available in the shader
    /// - content : a reference to the vector
    /// - usage : a description of how the buffer will be used in the shader
    /// - is_output : whether or not to retrieve the data in the buffer after execution
    pub fn create_buffer_from<T: ToVecU8<T>>(&mut self, name: &str, content: &Vec<T>, usage: BufferUsage, is_output: bool) {
        let (raw_content, data_type, data_type_stride, size) = <T as ToVecU8<T>>::convert(content);
        let byte_size = data_type_stride as u64 * size as u64;
        let id = self.buffers.len();
         // saves the position of the buffer if is_output: true, 
        // that is in order to iterates through output buffers
        if is_output {
            self.output_buffers_id.push(id);
        }
        // to be able to find the buffer from the name
        self.buf_name_to_id.insert(name.to_string(), id);
        // create a Buffer and adds it to the buffer vector
        self.buffers.push(Buffer {
            buffer: None,
            size: byte_size,
            name: name.to_owned(),
            type_: data_type,
            type_stride: data_type_stride,
            data: Some(raw_content),
            usage: usage.clone(),
            wgpu_usage: buffer_usage(usage, is_output),
            // if we need to get the output off the buffer, we add another one to copy to it
            is_output,
            output_buf: None
        });
    }

    /// Change for given buffer the usage. Is mainly used to add manualy a wgpu::BufferUsages.
    /// If you also want to change the easy_gpgpu::BufferUsage, you can do :
    /// wgpu::BuferUsages::some_usage | easy_gpgpu::BufferUsage::some_other_usage.
    pub fn apply_buffer_usages(&mut self, buf_name: &str, usage: wgpu::BufferUsages, is_output: bool) {
        let buf = &mut self.buffers[*self.buf_name_to_id.get(buf_name).expect("The buffer has not been created on this device (probably wrong name).")];
        buf.wgpu_usage |= usage;
        buf.is_output = is_output;
    }

    /// Apply a change over all elements of a vector.
    /// 
    /// Arguments :
    /// 
    /// - vec: the vector to apply the change on
    /// - code: what to do which each elements, a piece of wgsl shader
    /// in the shader, you have access to the `element` variable which represent an element in the vector
    /// that last line has to be the return value i.e :
    /// (more or less like in a rust function when we don't use return)
    /// ```
    /// let mut device = easy_gpgpu::Device::new();
    /// let v1 = vec![1u32, 2, 3];
    /// let v1 = device.apply_on_vector(v1, "element * 2u");
    /// assert_eq!(v1, vec![2u32, 4, 6]);
    /// 
    /// // this code will do for each element :
    /// // element = element * 2;
    /// ```
    /// 
    /// Important Note :
    /// There should be no variable/function/struct with a name containing the substring "element";
    /// 
    /// This function is limited to vectors with a byte length <= 1_073_741_824 (268_435_456 i32s max)
    /// 
    /// Possible problems you could encounter :
    /// 
    /// If you want to use this function with a previously created buffer, be aware that the number of buffers per execution is limited to 8 (you can have more but it's max 8 buffers used in the shader).
    /// This function creates automatically a certain number of buffers depending on the size (max size for a buffer is 134_217_728 bytes)
    pub fn apply_on_vector<T: ToVecU8<T> + std::clone::Clone>(&mut self, vec: Vec<T>, code: &str) -> Vec<T>{
        if vec.len() == 0 {
            return vec;
        }
        // remove all comments
        let code = remove_comments(code.to_string());
        let mut code = code.lines().collect::<Vec<_>>();
        // remove trailing whitespace
        while code.last().unwrap_or(&"// empty").chars().all(|e| " \t\r\n".contains(e)) {
            code.pop();
        }
        let mut code = code.join("\n");

        // finds the used buffers
        let mut used_buffers_id = vec![];
        for (i, b) in self.buffers.iter().enumerate() {
            let mut found = false;
            for c in IDENT_SEPARATOR_CHARS.chars() { // all the possible characters behind a variable name
                let patern = &format!("{c}{}[", b.name);
                if code.find(patern).is_some() {
                    found = true;
                }
            }
            // Adds the buffer to the bind_group for the compute pipeline later only if the buffer is used in the shader.
            if found {
                used_buffers_id.push(i);
            }
        }

        self.build_buffers(&used_buffers_id, true);

        let (_, type_, stride, size) = <T as ToVecU8<T>>::convert(&vec);
        // let other_code = code.lines().collect::<Vec<_>>().pop().unwrap();
        // let apply_expr = code.lines().last().unwrap();
        let byte_size =  stride as usize *size;
        let max_size = MAX_BYTE_BUFFER_SIZE / stride as usize;
        if byte_size <= MAX_BYTE_BUFFER_SIZE {
            code = code.replace("element", "reservedbuf[index]");
            let mut other_code = code.lines().collect::<Vec<_>>();
            other_code.pop();
            let other_code = other_code.join("\n");
            let other_code = other_code.as_str();
            let apply_expr = code.lines().last().unwrap();

            // adds the buffer to the buffer needed to be build
            used_buffers_id.push(self.buffers.len());
            // create the buffer
            self.create_buffer_from("reservedbuf", &vec, BufferUsage::ReadWrite, true);
            self.buffers[*self.buf_name_to_id.get("reservedbuf").unwrap()].wgpu_usage |= wgpu::BufferUsages::MAP_READ;
            // actualy builds the buffers
            self.build_buffers(&used_buffers_id, false);

            // execute the shader
            let shader_module = self.create_shader_module(Dispatch::Linear(vec.len()), &format!("
            {other_code}
            fn main() {{
                reservedbuf[index] = {apply_expr};
            }}
            ", ));
            self.execute_commands(vec![
                Command::Shader(shader_module),
                Command::Retrieve("reservedbuf")
            ]);
            let result = self.get_buffer_data(vec!["reservedbuf"]).into_iter().next().unwrap();
            return <T as ToVecU8<T>>::get_output(result);

        }else {
            // panics if used an index :
            for c in IDENT_SEPARATOR_CHARS.chars() {
                if code.contains(&(c.to_string()+"index")) {
                    panic!("Cannot use the `index` variable with a vector of more than {} bytes.", max_size * 4)
                }
            }

            let t = type_.to_string();
            let mut other_code = code.lines().collect::<Vec<_>>();
            other_code.pop();
            let other_code = other_code.join("\n");
            let other_code = other_code.as_str();

            let apply_expr = code.lines().last().unwrap();
            let function_element = format!("fn reservedfn(element: {t}) -> {t} {{return({apply_expr});}}\n");

            let nb_buf = (byte_size as f64 / MAX_BYTE_BUFFER_SIZE as f64).ceil() as u32;
            let size_last_buf = size % max_size;
            let mut main_body = vec![];

            // creates the buffer and generate the shader
            for i in 0..(nb_buf-1) {
                let vec_i = i as usize*max_size;
                self.create_buffer_from(&format!("reservedbuf{i}"), &vec[vec_i..(vec_i + max_size)].to_vec(), BufferUsage::ReadWrite, true);
                main_body.push(format!("reservedbuf{i}[index] = reservedfn(reservedbuf{i}[index]);\n"));
            }
            if size_last_buf != 0 {
                self.create_buffer_from(&format!("reservedbuf{}", nb_buf-1), &vec[(vec.len() - size_last_buf)..].to_vec(), BufferUsage::ReadWrite, true);
                main_body.push(format!("if (index < {size_last_buf}u) {{ reservedbuf{0}[index] = reservedfn(reservedbuf{0}[index]);}}\n", nb_buf-1))
            }

            // execute the shader
            let shader_module = self.create_shader_module(Dispatch::Linear(max_size), &format!("
            {other_code}
            {function_element}
            fn main() {{
                {}
            }}
            ", main_body.join("")));
            let mut commands = vec![Command::Shader(shader_module)];
            let mut buffer_names = vec![];
            for i in 0..nb_buf {
                buffer_names.push(format!("reservedbuf{i}"));
            }
            for name in buffer_names.iter() {
                commands.push(Command::Retrieve(&name));
            }
            self.execute_commands(commands);
            let mut bufs = vec![];
            for i in 0..nb_buf {
                bufs.push(format!("reservedbuf{i}"));
            }
            let results = self.get_buffer_data(bufs.iter().map(|e| e.as_str()).collect::<Vec<_>>());
            let mut out = Vec::<T>::new();
            for result in results {
                out.append(&mut <T as ToVecU8<T>>::get_output(result));
            }
            return out;
        }
    }

    /// Creates a shader module to execute it several time on the device
    /// If you don't need to use to execute the shader several time, use `execute_shader_code` instead.
    #[inline]
    pub fn create_shader_module(&self, dispatch: Dispatch, code: &str) -> ShaderModule {
        let dispatch_linear_len;
        match dispatch {
            Dispatch::Linear(l) => {
                dispatch_linear_len = Some(l);
            }
            Dispatch::Custom(_, _, _) => {
                dispatch_linear_len = None;
            } 
        }
        let dispatch_linear = dispatch_linear_len.is_some();

        let mut tmp_code = remove_comments(code.to_owned());
        // adds stage and workgroup attributes and gets the global id (required by wgsl language)
        let mut main_headers = String::from("
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {\n");
        // derive the index from the global_id
        if dispatch_linear {
            if dispatch_linear_len < Some(65536) {
                main_headers += "\tlet index: u32 = global_id.x;\n";
            }else {
                main_headers += &format!("\tlet index: u32 = global_id.x + global_id.y * 65535u;\nif (index >= {}u) {{return;}}\n", dispatch_linear_len.unwrap());
                
            }
        }

        // find the used buffers
        let mut used_buffers_id = vec![];
        // replaces buf[index] with buf.d[index] because the data types are structs
        for (i, b) in self.buffers.iter().enumerate() {
            let mut found = false;
            for c in IDENT_SEPARATOR_CHARS.chars() { // all the possible characters behind a variable name
                let patern = &format!("{c}{}[", b.name);
                if tmp_code.find(patern).is_some() {
                    found = true;
                }
                tmp_code = tmp_code.replace(patern, &format!("{c}{}.d[", b.name));
            }

            // Adds the buffer to the bind_group for the compute pipeline later only if the buffer is used in the shader.
            if found {
                used_buffers_id.push(i);
            }
        }

        let mut structs = vec![];
        let mut struct_types = HashMap::new();
        let mut bindings = vec![];
        let mut used_buffers_id_i = 0;
        let mut binding_i = 0;
        for (i, b) in self.buffers.iter().enumerate() {
            while used_buffers_id_i < used_buffers_id.len()-1 && used_buffers_id[used_buffers_id_i] < i {
                used_buffers_id_i += 1;
            }
            if used_buffers_id[used_buffers_id_i] != i {
                continue;
            }
            // if b.type_stride == -1 : the type is manually coded
            // checks if the type is in the struct_types to prevent code duplication in the shader
            if b.type_stride != -1 && !struct_types.contains_key(&b.type_){
                structs.push(format!("struct reserved{i} {{\n\td: [[stride({})]] array<{}>;\n}};\n",b.type_stride, b.type_.to_string()));
                struct_types.insert(b.type_.clone(), i);
            }
            
            // gets the buffer from bindings to make it available in the shader (this is some more wgsl staf)
            bindings.push(format!(
                "[[group(0), binding({binding_i})]] \n var<storage, {}> {}: reserved{};\n",
                match b.usage {
                    BufferUsage::ReadOnly => {"read".to_string()},
                    BufferUsage::WriteOnly => {"write".to_string()},
                    BufferUsage::ReadWrite => {"read_write".to_string()}
                },
                b.name,
                struct_types.get(&b.type_).unwrap()
            ));

            binding_i += 1;
        }
        
        // put structs, bindings, main_headers in the shader
        tmp_code = format!("{}{}{}",structs.join(""), bindings.join(""),  tmp_code.replace("fn main() {\n", &main_headers));

        ShaderModule {
            shader: self.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&tmp_code))
            }),
            // bind_group: bind_group_entries
            used_buffers_id,
            dispatch
        }
    }

    /// Builds the buffers at the specified index and the buffer with `.is_output: true`
    fn build_buffers(&mut self, indices: &Vec<usize>, build_output: bool) { // build the buffer with the id specified + all the output buffers
        let mut bufs_to_build = vec![];
        let mut indices_i = 0;
        if indices.len() == 0 {
            for b in self.buffers.iter_mut() {
                if b.is_output { // easier b.is_output is true or the index of b in is ids.
                    bufs_to_build.push(b);
                }
            }
        }else if build_output {
            for (i, b) in self.buffers.iter_mut().enumerate() {
                while indices_i < indices.len()-1 && indices[indices_i] < i {
                    indices_i += 1;
                }
                if b.is_output || i == indices[indices_i] { // easier b.is_output is true or the index of b in is ids.
                    bufs_to_build.push(b);
                }
            }
        }else {
            for (i, b) in self.buffers.iter_mut().enumerate() {
                while indices_i < indices.len()-1 && indices[indices_i] < i {
                    indices_i += 1;
                }
                if i == indices[indices_i] { // easier b.is_output is true or the index of b in is ids.
                    bufs_to_build.push(b);
                }
            }
        }
        

        for buf in bufs_to_build.iter_mut() {
            // if the buffer has already been built, we don't rebuild it
            if buf.buffer.is_some(){
                continue;
            }
            match &buf.data {
                Some(data) => {
                    buf.buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&buf.name),
                        contents: &data,
                        usage: buf.wgpu_usage,
                    }));
                }
                None => {
                    buf.buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some(&buf.name),
                            size: buf.size,
                            usage: buf.wgpu_usage,
                            mapped_at_creation: false,
                        }))
                }
            }
            if buf.is_output {
                buf.output_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&buf.name),
                    size: buf.size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }))
            }
        }
    }

    /// Executes a shader module on the gpu which can be create with `device.create_shader_module`
    /// 
    /// Arguments :
    /// 
    /// - shader_module : a compiled shader
    /// 
    /// Return value (same as the `execute_shader_code` function):
    /// 
    /// The output is a vector of vector : a liste of all the buffers outputed, to get the raw values, call for example `output[0].unwrap_i32()`to get a `Vec<i32>`.
    /// (there is also unwrap_u32, unwrap_f32 and unwrap_bool)
    /// It is advised to turn this into an iterator using : output.into_iter() and get the outputed vectors with the next() method.
    /// 
    /// Panics :
    /// 
    /// - if there has not been any buffer created before the execution
    /// 
    /// Note that if you want to execute the shader only once you can do two steps at once with `device.execute_shader_code`.
    #[inline]
    pub fn execute_shader_module(&mut self, shader_module: &ShaderModule) -> Vec<OutputVec> {
        if self.buffers.len() == 0 {
            panic!("In function `Device::execute_shader_module` : Cannot execute a shader if no buffer has been created. 
            You can create a buffer using the `Device::create_buffer` and `Device::create_buffer_from` functions");
        }

        self.build_buffers(&shader_module.used_buffers_id, false);

        let mut bind_group_entries = vec![];
        let mut i = 0;
        for id in shader_module.used_buffers_id.iter() {
            bind_group_entries.push(wgpu::BindGroupEntry{
                binding: i as u32,
                resource: self.buffers[*id].buffer.as_ref().unwrap().as_entire_binding()
            });
            i += 1;
        }

        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader_module.shader,
            entry_point: "main",
        });

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &bind_group_entries
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            match shader_module.dispatch {
                Dispatch::Linear(val) => {
                    if val < 65536 {
                        cpass.dispatch(val as u32, 1, 1);
                    }else {
                        cpass.dispatch(65535, (val as f64 / 65535f64).ceil() as u32, 1);
                    }
                }
                Dispatch::Custom(x, y, z) => {
                    cpass.dispatch(x, y, z);
                }
            }
        }
        // adds the copy to retrieve the output from the buffers with is_output = true
            for i in shader_module.used_buffers_id.iter() {
                let b = &self.buffers[*i];
                if b.is_output {
                    encoder.copy_buffer_to_buffer(&b.buffer.as_ref().unwrap(), 0, &b.output_buf.as_ref().unwrap(), 0, b.size);
                }
            }
        self.queue.submit(Some(encoder.finish()));
        
        self.retrieve_buffer_data(&shader_module.used_buffers_id)
    }

    /// The single most important method in this crate : it runs the wgsl shader code on the gpu and gets the output
    /// 
    /// Arguments :
    /// 
    /// - dispatch : most interesting use is `Dispatch::Linear(n)`, it will add a variable in the shader, `index` which varies from 0 to n over all executions of the shader
    /// if `Dispatch::Custom(x, y, z)` is used, you will get the global_id variable with three fields : `global_id.x`, `global_id.y` and `global_id.z`
    /// - code : the shader code to run on the gpu, the buffers created before on this device are available with the name provided
    /// 
    /// Output :
    /// 
    /// The output is a vector of vector : a liste of all the buffers outputed, to get the raw values, call for example `output[0].unwrap_i32()`to get a `Vec<i32>`.
    /// (there is also unwrap_u32, unwrap_f32 and unwrap_bool)
    /// It is advised to turn this into an iterator using : output.into_iter() and get the outputed vectors with the next() method.
    /// 
    /// Important note :
    /// 
    /// The first line of your shader should be exactly : "fn main() {", the shader will not get preprocessed correctly otherwise.
    pub fn execute_shader_code(&mut self, dispatch: Dispatch, code: &str) -> Vec<OutputVec>{
        let shader_module = self.create_shader_module(dispatch.clone(), code);
        self.execute_shader_module(&shader_module)
    }

    /// Execute a list of commands, there can be at most a single shader command.
    pub fn execute_commands(&mut self, commands: Vec<Command>) {
        let mut shader_count = 0;
        let mut shader_index = None;
        for (i, c) in commands.iter().enumerate() {
            if let Command::Shader(_) = c {
                shader_count += 1;
                shader_index = Some(i);
            }
        }
        if shader_count > 1 {
            panic!("In function `Device::execute_commands` : There should only be 1 shader in the commands, got {shader_count}");
        }

        // adds the wgpu::BufferUsages needed to copy
        let mut copy_buffers = HashSet::new();
        for c in commands.iter() {
            if let Command::Copy(from, to) = c {
                let buf_id1 = self.buf_name_to_id.get(&from.to_string()).expect("The source buffer has not been created on this device (probably wrong name).");
                self.buffers[*buf_id1].wgpu_usage |= wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_READ;
                let buf_id2 = self.buf_name_to_id.get(&to.to_string()).expect("The destination buffer has not been created on this device (probably wrong name).");
                self.buffers[*buf_id2].wgpu_usage |= wgpu::BufferUsages::COPY_DST;
                copy_buffers.insert(*buf_id1);
                copy_buffers.insert(*buf_id2);
            }
        }

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        if shader_count == 0 { // the commands are just copies between buffers
            for c in commands.iter() {
                if let Command::Copy(from, to) = c {
                    let buf1 = &self.buffers[*self.buf_name_to_id.get(&from.to_string()).unwrap()]; // a easy_gpgpu::Buffer (to take the size)
                    let buf2 = &self.buffers[*self.buf_name_to_id.get(&to.to_string()).unwrap()].buffer; // a wgpu::Buffer
                    encoder.copy_buffer_to_buffer(&buf1.buffer.as_ref().unwrap(), 0, &buf2.as_ref().unwrap(), 0, buf1.size);
                }
            }
        }else { // there is a shader in the commands
            let shader_index = shader_index.unwrap();
            for c in commands.iter().take(shader_index) {
                if let Command::Copy(from, to) = c {
                    let buf1 = &self.buffers[*self.buf_name_to_id.get(&from.to_string()).unwrap()];
                    let buf2 = &self.buffers[*self.buf_name_to_id.get(&to.to_string()).unwrap()].buffer;
                    encoder.copy_buffer_to_buffer(&buf1.buffer.as_ref().unwrap(), 0, &buf2.as_ref().unwrap(), 0, buf1.size);
                }
            }

            let shader_module = 
            (if let Command::Shader(sm) = &commands[shader_index] {
                Some(sm)
            }else {
                None
            }).unwrap();

            let used_buffers_shader_and_copy = &shader_module.used_buffers_id.iter().map(|e| *e).chain(copy_buffers.into_iter()).collect::<Vec<_>>();
            self.build_buffers(used_buffers_shader_and_copy, true);

            let mut bind_group_entries = vec![];
            let mut i = 0;
            for id in shader_module.used_buffers_id.iter() {
                bind_group_entries.push(wgpu::BindGroupEntry{
                    binding: i as u32,
                    resource: self.buffers[*id].buffer.as_ref().unwrap().as_entire_binding()
                });
                i += 1;
            }

            let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader_module.shader,
                entry_point: "main",
            });

            let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &bind_group_entries
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("execute commands")});
                cpass.set_pipeline(&compute_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                match shader_module.dispatch {
                    Dispatch::Linear(val) => {
                        if val < 65536 {
                            cpass.dispatch(val as u32, 1, 1);
                        }else {
                            cpass.dispatch(65535, (val as f64 / 65535f64).ceil() as u32, 1);
                        }
                    }
                    Dispatch::Custom(x, y, z) => {
                        cpass.dispatch(x, y, z);
                    }
                }
            }

            // adds the copy commands that come after the shader
            for c in commands.iter().skip(shader_index+1) {
                if let Command::Copy(from, to) = c {
                    let buf1 = &self.buffers[*self.buf_name_to_id.get(&from.to_string()).unwrap()];
                    let buf2 = &self.buffers[*self.buf_name_to_id.get(&to.to_string()).unwrap()].buffer;
                    encoder.copy_buffer_to_buffer(&buf1.buffer.as_ref().unwrap(), 0, &buf2.as_ref().unwrap(), 0, buf1.size);
                }
                if let Command::Retrieve(buf) = c {
                    let buf1 = &self.buffers[*self.buf_name_to_id.get(&buf.to_string()).unwrap()];
                    encoder.copy_buffer_to_buffer(&buf1.buffer.as_ref().unwrap(), 0, &buf1.output_buf.as_ref().unwrap(), 0, buf1.size);
                }
            }

            self.queue.submit(Some(encoder.finish()));
            
    
        }
    }

    #[inline]
    fn retrieve_buffer_data(&self, buffers_id: &Vec<usize>) -> Vec<OutputVec> {
        let mut output_buffers_id = vec![];
        let mut buffer_slices = vec![];
        let mut result_futures = vec![];
        for i in buffers_id.iter() {
            let b = &self.buffers[*i];
            if !b.is_output {
                continue;
            }
            let out_b = b.output_buf.as_ref().unwrap();
            // b.is_output_buf_mapped = true;
            let buffer_slice = out_b.slice(..);
            let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
            buffer_slices.push(buffer_slice);
            result_futures.push(buffer_future);
            output_buffers_id.push(*i);
        }

        // Poll the device in a blocking manner so that our future resolves.
        self.device.poll(wgpu::Maintain::Wait);


        let mut results = Vec::with_capacity(self.output_buffers_id.len());
        // stores the index of the coresponding buffer (to the output buffer)
        let mut buffer_index = 0;

        async {
            for (i, result_future) in result_futures.into_iter().enumerate() {
                let buf = &self.buffers[output_buffers_id[i]];
                buffer_index += 1;
                if let Ok(()) = result_future.await {
                    // Gets contents of buffer
                    let data = buffer_slices[i].get_mapped_range();
                    // Since contents are in bytes, this converts these bytes back to u32
                    let result: OutputVec;
                    match buf.type_ {
                        BufferType::I32 => {
                            result = OutputVec::VecI32(bytemuck::cast_slice::<u8, i32>(&data).to_vec());
                        },
                        BufferType::U32 => {
                            result = OutputVec::VecU32(bytemuck::cast_slice::<u8, u32>(&data).to_vec());
                        },
                        BufferType::F32 => {
                            result = OutputVec::VecF32(bytemuck::cast_slice::<u8, f32>(&data).to_vec());
                        },
                        BufferType::Bool => {
                            result = OutputVec::VecBool(data.iter().map(|&e| e != 0).collect());
                        }
                    }

                    // With the current wgpu interface, we have to make sure all mapped views are dropped before we unmap the buffer.
                    drop(data);
                    buf.output_buf.as_ref().unwrap().unmap(); // Unmaps buffer from memory
                    
                    // Returns data from buffer
                    results.push(result);
                }else {
                    panic!("computations failed");
                }
            }
        results
        }.block_on()
    }

    /// Retrieves the data of buffers to vectors. It has to have been created with `is_output: true`.
    pub fn get_buffer_data(&self, buffer_names: Vec<&str>) -> Vec<OutputVec> {
        let buffers_id = buffer_names.iter().map(|&e| *self.buf_name_to_id.get(e)
            .expect("The buffer has not been created on this device (probably wrong name)"))
            .collect::<Vec<_>>();
        self.retrieve_buffer_data(&buffers_id)
    }
}

pub mod examples {
    use crate::{Device, BufferUsage, Dispatch, BufferType, Command};

    /// The simplest method to use
    pub fn simplest_apply() {
        let mut device = Device::new();
        let v1 = vec![1.0f32, 2.0, 3.0];
        // the next line reads : for every element in v1, perform : element = element * 2.0
        let v1 = device.apply_on_vector(v1, "element * 2.0");
        println!("{v1:?}");
    }

    /// An example of device.apply_on_vector with a previously created buffer
    pub fn apply_with_buf() {
        let mut device = Device::new();
        let v1 = vec![2.0f32, 3.0, 5.0, 7.0, 11.0];
        let exponent = vec![3.0];
        device.create_buffer_from("exponent", &exponent, BufferUsage::ReadOnly, false);
        let cubes = device.apply_on_vector(v1, "pow(element, exponent[0u])");
        println!("{cubes:?}")
    }

    /// The simplest example with device.execute_shader_code : multiplying by 2 every element of a vector.
    pub fn with_execute_shader() {
        let mut device = Device::new();
        let v1 = vec![1i32, 2, 3, 4, 5, 6];
        device.create_buffer_from("v1", &v1, BufferUsage::ReadOnly, false);
        device.create_buffer("output", BufferType::I32, v1.len(), BufferUsage::WriteOnly, true);
        let result = device.execute_shader_code(Dispatch::Linear(v1.len()), r"
        fn main() {
            output[index] = v1[index] * 2;
        }
        ").into_iter().next().unwrap().unwrap_i32();
        assert_eq!(result, vec![2, 4, 6, 8, 10, 12]);
    }

    /// An example with multiple returned buffer with device.execute_shader_code
    pub fn multiple_output_buffers() {
        let mut device = Device::new();
        let v = vec![1u32, 2, 3];
        let v2 = vec![3u32, 4, 5];
        let v3 = vec![7u32, 8, 9];
        // we specify is_output: true so this will be our first returned buffer
        device.create_buffer_from(
            "buf",
            &v,
            BufferUsage::ReadWrite,
            true
        );
        device.create_buffer_from(
            "buf2",
            &v2,
            BufferUsage::ReadOnly,
            false
        );
        // we specify is_output: true so this will be our second returned buffer
        device.create_buffer_from(
            "buf3",
            &v3,
            BufferUsage::ReadWrite,
            true
        );
        let mut result = device.execute_shader_code(Dispatch::Linear(v.len()), r"
            fn main() {
                buf[index] = buf[index] + buf2[index] + buf3[index];
                buf3[index] = buf[index] * buf2[index] * buf3[index];
            }
        ").into_iter();

        let sum = result.next().unwrap().unwrap_u32();
        let product = result.next().unwrap().unwrap_u32();
        println!("{:?}", sum);
        println!("{:?}", product);
    }

    /// An Example with a custom dispatch that gives access to the global_id variable.
    pub fn global_id() {
        let mut device = Device::new();
        let vec = vec![2u32, 3, 5, 7, 11, 13, 17];
        // "vec" is actually a reserved keyword in wgsl.
        device.create_buffer_from("vec1", &vec, BufferUsage::ReadWrite, true);
        let result = device.execute_shader_code(Dispatch::Custom(1, vec.len() as u32, 1), r"
        fn main() {
            vec1[global_id.y] = vec1[global_id.y] + global_id.x + global_id.z;
        }
        ").into_iter().next().unwrap().unwrap_u32();
        // since the Dispatch was (1, 7, 1), the global_id.x and global.y are always 0
        // so our primes stay primes :
        assert_eq!(result, vec![2u32, 3, 5, 7, 11, 13, 17]);
    }

    /// An example where we create a shader_module and then execute it (in practice it is for reusing the shader_module several times).
    pub fn shader_two_steps() {
        let mut device = Device::new();
        let v = vec![1u32, 2, 3, 4];
        device.create_buffer_from("buf1", &v, BufferUsage::ReadWrite, true);
        let shader_module = device.create_shader_module(Dispatch::Linear(v.len()), "
        fn main() {
            buf1[index] = buf1[index] * 17u;
        }
        ");
        let result = device.execute_shader_module(&shader_module).into_iter().next().unwrap().unwrap_u32();
        assert_eq!(result, vec![17u32, 34, 51, 68]);
    }

    /// An example with a complete pipeline which as you can see, is quite annoying just to multiply a vector by 2.
    pub fn complete_pipeline() {
        let mut device = Device::new();
        let v1 = vec![1u32, 2, 3, 4, 5];
        device.create_buffer_from("v1", &v1, BufferUsage::ReadWrite, true);
        let shader_module = device.create_shader_module(Dispatch::Linear(v1.len()), "
        fn main() {
            v1[index] = v1[index] * 2u;
        }
        ");
        let mut commands = vec![];
        commands.push(Command::Shader(shader_module));
        commands.push(Command::Retrieve("v1"));
        device.execute_commands(commands);
        let result = device.get_buffer_data(vec!["v1"]).into_iter().next().unwrap().unwrap_u32();
        assert_eq!(result, vec![2u32, 4, 6, 8, 10]);
    }

    /// An example where we execute two shaders on the same device, with the same buffers.
    pub fn reusing_device() {
        let mut device = Device::new();
        let v1 = vec![1i32, 2, 3, 4, 5, 6];
        device.create_buffer_from("v1", &v1, BufferUsage::ReadOnly, false);
        device.create_buffer("output", BufferType::I32, v1.len(), BufferUsage::WriteOnly, true);
        let result = device.execute_shader_code(Dispatch::Linear(v1.len()), r"
        fn main() {
            output[index] = v1[index] * 2;
        }
        ").into_iter().next().unwrap().unwrap_i32();
        assert_eq!(result, vec![2i32, 4, 6, 8, 10, 12]);

        let result2 = device.execute_shader_code(Dispatch::Linear(v1.len()), r"
        fn main() {
            output[index] = v1[index] * 10;
        }
        ").into_iter().next().unwrap().unwrap_i32();
        assert_eq!(result2, vec![10, 20, 30, 40, 50, 60]);
    }

    /// An example of with device.execute_shader_code with a large scale computation.
    #[test]
    pub fn big_computations() {
        let mut device = Device::new();
        let size = 33_554_432;
        device.create_buffer(
            "buf",
            BufferType::U32,
            size,
            BufferUsage::WriteOnly,
            true);
        let result = device.execute_shader_code(Dispatch::Linear(size), r"
        fn number_of_seven_in_digit_product(number: u32) -> u32 {
            var p: u32 = 1u;
            var n: u32 = number;
            loop {
                if (n == 0u) {break;}
                p = p * (n % 10u);
                n = n / 10u;
            }
            var nb_seven: u32 = 0u;
            loop {
                if (p == 0u) {break;}
                if (p % 10u == 7u) {
                    nb_seven = nb_seven + 1u;
                }
                p = p / 10u;
            }
            return nb_seven;
        }
        fn main() {
            buf[index] = number_of_seven_in_digit_product(index);
        }
        ").into_iter().next().unwrap().unwrap_u32();
        let mut index = 0;
        let mut max = result[0];
        for (i, e) in result.iter().enumerate() {
            if e > &max {
                max = *e;
                index = i;
            }
        }
        println!("The number who's digit product got the most seven below {size} is {index} with {max} sevens.");
    }
}
