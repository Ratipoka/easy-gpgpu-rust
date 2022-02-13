//! # Easy GPGPU
//! A high level, easy to use async gpgpu crate based on [`wgpu`](https://github.com/gfx-rs/wgpu).
//! It is made for very large computations on powerful gpus
//! 
//! Main goals :
//! 
//! - make general purpose computing very simple
//! - make it as easy as possible to write wgsl shaders
//! - deal with binding buffers automatically
//! 
//! Limitations :
//! 
//! - only types available for buffers : bool, i32, u32, f32
//! - max buffer byte_size : around 134_000_000 (~33 million i32)
//! - depending on the driver, a process will be killed if it takes more than 3 seconds on the gpu
//! - takes a bit of time to initiate the device (due to wgpu backends)
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
//!     device.create_buffer_from("inputs", &v, BufferUsage::ReadWrite, true);
//!     let result = device.execute_shader_code(Dispatch::Linear(v.len()), r"
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
//!         }
//!         return i;
//!     }
//! 
//!     fn main() {
//!         inputs[index] = collatz_iterations(inputs[index]);
//!     }"
//!     ).into_iter().next().unwrap().unwrap_u32();
//!     assert_eq!(result, vec![0, 2, 7, 55]);
//! }
//! ```
//! => No binding, no annoying global_id, no need to use a low level api.
//! You just declare the name of the buffer and it is immediately available in the wgsl shader.
//! 
//! ## Usage 
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

use std::{vec, collections::HashMap};
use wgpu::util::DeviceExt;
use std::vec::Vec;
use pollster::FutureExt;

/// A buffer representation build on top of the `wgpu::Buffer`.
/// It stores a buffer name to enable a direct access in the shader.
/// To get a buffer, call `device.create_buffer` or `device.create_buffer_from`.
struct Buffer {
    buffer: wgpu::Buffer,
    name: String,
    size: u64,
    type_: BufferType,
    type_stride: i32,
    usage: BufferUsage,
    is_output: bool,
}

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
pub enum Dispatch {
    Linear(usize),
    Custom(u32, u32, u32)
}

/// Trait to obtain raw data of vectors in order to convert them to buffer as seamlessly as possible
pub trait ToVecU8<T> {
    // returns raw content, type, type_stride, length of vec
    fn convert(v: T) -> (Vec<u8>, BufferType, i32, usize);
}
impl ToVecU8<&Vec<i32>> for &Vec<i32> {
    fn convert(v: &Vec<i32>) -> (Vec<u8>, BufferType, i32, usize) {
        (bytemuck::cast_slice::<i32, u8>(&v).to_vec(), BufferType::I32, 4, v.len())
    }
}
impl ToVecU8<&Vec<u32>> for &Vec<u32> {
    fn convert(v: &Vec<u32>) -> (Vec<u8>, BufferType, i32, usize) {
        (bytemuck::cast_slice::<u32, u8>(&v).to_vec(), BufferType::U32, 4, v.len())
    }
}
impl ToVecU8<&Vec<f32>> for &Vec<f32> {
    fn convert(v: &Vec<f32>) -> (Vec<u8>, BufferType, i32, usize) {
        (bytemuck::cast_slice::<f32, u8>(&v).to_vec(), BufferType::F32, 4, v.len())
    }
}
impl ToVecU8<&Vec<bool>> for Vec<bool> {
    fn convert(v: &Vec<bool>) -> (Vec<u8>, BufferType, i32, usize) {
        (v.iter().map(|&e| e as u8).collect::<Vec<_>>(), BufferType::Bool, 1, v.len())
    }
}

/// An enum to represent the different output vectors possible
/// To get the vector back, call .unwrap_i32() for example.
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
                panic!("value is not a u32!");
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

/// The main struct which provides abstraction over the wgpu library.
pub struct Device {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: Vec<Buffer>,
    output_buffers: Vec<wgpu::Buffer>
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
                output_buffers: vec![]
            }
        }.block_on()
        
    }

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
        self.buffers.push(Buffer {
            buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: byte_size as u64,
                usage: buffer_usage(usage.clone(), is_output),
                mapped_at_creation: false,
            }),
            name: name.to_owned(),
            size: byte_size as u64,
            type_: data_type,
            type_stride: data_type_stride as i32,
            usage,
            is_output
        });
        // if we need to get the output off the buffer, we add another one to copy to
        if is_output {
            self.output_buffers.push(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: byte_size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
    }

    /// Creates a buffer from a vector.
    /// 
    /// Arguments :
    /// 
    /// - name : the name of the buffer that will be available in the shader
    /// - content : a reference to the vector
    /// - usage : a description of how the buffer will be used in the shader
    /// - is_output : whether or not to retrieve the data in the buffer after execution
    pub fn create_buffer_from<T: ToVecU8<T>>(&mut self, name: &str, content: T, usage: BufferUsage, is_output: bool) {
        let (raw_content, data_type, data_type_stride, size) = <T as ToVecU8<T>>::convert(content);
        let byte_size = data_type_stride as u64 * size as u64;
        self.buffers.push(Buffer {
            buffer: self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: &raw_content,
                usage: buffer_usage(usage.clone(), is_output)
            }),
            size: byte_size,
            name: name.to_owned(),
            type_: data_type,
            type_stride: data_type_stride,
            usage,
            is_output
        });
        // if we need to get the output off the buffer, we add another one to copy to
        if is_output {
            self.output_buffers.push(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
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
    /// The output is a vector of vector : a liste of all the buffers outputed, to get the raw values, call for example `output[0].unwrap_i32()`to get a `Vec<i32>` .
    /// It is advised to turn this into an iterator using : output.into_iter() and get the outputed vectors with the next() method.
    /// 
    /// Important note :
    /// 
    /// The first line of your shader should be exactly : "fn main() {", the shader will not get preprocessed correctly otherwise.
    pub fn execute_shader_code(&mut self, dispatch: Dispatch, code: &str) -> Vec<OutputVec>{
        // this whole function is divided into 2 parts : the preprocessing of the shader and bindgroup setup and the boring part of the wgpu api
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

        let mut tmp_code = code.to_owned();
        // adds stage and workgroup attributes and gets the global id (required by wgsl language)
        let mut main_headers = String::from("
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {\n");
        // derive the index from the global_id
        if dispatch_linear {
            if dispatch_linear_len < Some(65536) {
                main_headers += "\tlet index: u32 = global_id.x;";
            }else {
                main_headers += &format!("\tlet index: u32 = global_id.x + global_id.y * 65535u;\nif (index >= {}u) {{return;}}\n", dispatch_linear_len.unwrap());
                
            }
        }

        let mut structs = vec![];
        let mut struct_types = HashMap::new();
        let mut bindings = vec![];
        let mut bind_group_entries = vec![];
        for (i, b) in self.buffers.iter().enumerate() {
            // if b.type_stride == -1 : the type is manually coded
            // checks if the type is in the struct_types to prevent code duplication in the shader
            if b.type_stride != -1 && !struct_types.contains_key(&b.type_){
                structs.push(format!("struct reserved{i} {{\n\td: [[stride({})]] array<{}>;\n}};\n",b.type_stride, b.type_.to_string()));
                struct_types.insert(b.type_.clone(), i);
            }
            
            // gets the buffer from bindings to make it available in the shader (this is some more wgsl staf)
            bindings.push(format!(
                "[[group(0), binding({i})]] \n var<storage, {}> {}: reserved{};\n",
                match b.usage {
                    BufferUsage::ReadOnly => {"read".to_string()},
                    BufferUsage::WriteOnly => {"write".to_string()},
                    BufferUsage::ReadWrite => {"read_write".to_string()}
                },
                b.name,
                struct_types.get(&b.type_).unwrap()
            ));

            // adds the buffer to the bind_group for the compute pipeline later
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.buffer.as_entire_binding(),
            })
        }

        // put structs, bindings, main_headers in the shader
        tmp_code = tmp_code.replace("fn main() {\n", &format!("{}{}{}", structs.join(""), bindings.join(""), main_headers));
        // replaces buf[index] with buf.d[index] because the data types are structs
        for b in self.buffers.iter() {
            for c in " \t\n\r([,+-/*=%&|^~!<>{}".chars() { // all the possible characters behind a variable name
                tmp_code = tmp_code.replace(&format!("{c}{}[", b.name), &format!("{c}{}.d[", b.name));
            }
        }

        // boring api staff :

        let cs_module = self.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&tmp_code)),
        });

        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &cs_module,
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
            match dispatch {
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
        let mut output_buf_i = 0;
            for b in self.buffers.iter() {
                if b.is_output {
                    encoder.copy_buffer_to_buffer(&b.buffer, 0, &self.output_buffers[output_buf_i], 0, b.size);
                    output_buf_i += 1;
                }
            }
        self.queue.submit(Some(encoder.finish()));
        

        let mut buffer_slices = vec![];
        let mut result_futures = vec![];
        for out_b in self.output_buffers.iter() {
            let buffer_slice = out_b.slice(..);
            let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
            buffer_slices.push(buffer_slice);
            result_futures.push(buffer_future);
        }

        // Poll the device in a blocking manner so that our future resolves.
        self.device.poll(wgpu::Maintain::Wait);


        let mut results = Vec::with_capacity(self.output_buffers.len());
        // stores the index of the coresponding buffer (to the output buffer)
        let mut buffer_index = 0;
        async {
            for (i, result_future) in result_futures.into_iter().enumerate() {
                while !self.buffers[buffer_index].is_output {
                    buffer_index += 1;
                }
                if let Ok(()) = result_future.await {
                    // Gets contents of buffer
                    let data = buffer_slices[i].get_mapped_range();
                    // Since contents are in bytes, this converts these bytes back to u32
                    let result: OutputVec;
                    match self.buffers[buffer_index].type_ {
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
            
                    // With the current interface, we have to make sure all mapped views are
                    // dropped before we unmap the buffer.
                    drop(data);
                    self.output_buffers[i].unmap(); // Unmaps buffer from memory
            
                    // Returns data from buffer
                    results.push(result);
                }else {
                    panic!("computations failed");
                }
            }
        results
        }.block_on()
    }
}

pub mod examples {
    use crate::{Device, BufferUsage, Dispatch, BufferType};

    /// The simplest example : multiplying by 2 every element of a vector.
    pub fn example1() {
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

    /// An example with multiple returned buffer
    pub fn example2() {
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

    /// The comparison with wgpu hello-compute : we go from 205 significant lines of code to only 34
    pub fn wgpu_hello_compute() {
        let mut device = Device::new();
        let v = vec![1u32, 4, 3, 295];
        device.create_buffer_from("inputs", &v, BufferUsage::ReadWrite, true);
        let result = device.execute_shader_code(Dispatch::Linear(v.len()), r"
        fn collatz_iterations(n_base: u32) -> u32{
            var n: u32 = n_base;
            var i: u32 = 0u;
            loop {
                if (n <= 1u) {
                    break;
                }
                if (n % 2u == 0u) {
                    n = n / 2u;
                }
                else {
                    // Overflow? (i.e. 3*n + 1 > 0xffffffffu?)
                    if (n >= 1431655765u) {   // 0x55555555u
                        return 4294967295u;   // 0xffffffffu
                    }
        
                    n = 3u * n + 1u;
                }
                i = i + 1u;
            }
            return i;
        }
        fn main() {
            inputs[index] = collatz_iterations(inputs[index]);
        }
        ").into_iter().next().unwrap().unwrap_u32();
        assert_eq!(result, vec![0, 2, 7, 55]);
    }

    /// An example where we execute two shaders on the same device, with the same buffers.
    pub fn reusing_device() {
        // example 1
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

    /// An example of using a very big buffer :
    pub fn big_computations() {
        let mut device = Device::new();
        let size = 30_000_000;
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
        let mut max = &result[0];
        let mut index = 0;
        for (i, e) in result.iter().enumerate() {
            if e > max {
                max = e;
                index = i;
            }
        }
        println!("The number with the most sevens in the product of it's digits (below {size}) is {}", index);
    }
}