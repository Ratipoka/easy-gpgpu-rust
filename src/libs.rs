use std::{time::Instant, vec, collections::HashMap};
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
    type_: String,
    type_stride: i32,
    usage: BufferUsage,
    is_output: bool,
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
    fn convert(v: T) -> (Vec<u8>, String, i32, usize);
}
impl ToVecU8<&Vec<i32>> for &Vec<i32> {
    fn convert(v: &Vec<i32>) -> (Vec<u8>, String, i32, usize) {
        (bytemuck::cast_slice::<i32, u8>(&v).to_vec(), String::from("i32"), 4, v.len())
    }
}
impl ToVecU8<&Vec<u32>> for &Vec<u32> {
    fn convert(v: &Vec<u32>) -> (Vec<u8>, String, i32, usize) {
        (bytemuck::cast_slice::<u32, u8>(&v).to_vec(), String::from("u32"), 4, v.len())
    }
}
impl ToVecU8<&Vec<f32>> for &Vec<f32> {
    fn convert(v: &Vec<f32>) -> (Vec<u8>, String, i32, usize) {
        (bytemuck::cast_slice::<f32, u8>(&v).to_vec(), String::from("f32"), 4, v.len())
    }
}
impl ToVecU8<&Vec<bool>> for Vec<bool> {
    fn convert(v: &Vec<bool>) -> (Vec<u8>, String, i32, usize) {
        (v.iter().map(|&e| e as u8).collect::<Vec<_>>(), String::from("bool"), 1, v.len())
    }
}

/// An enum to represent the different output vectors possible
/// To get the vector back, call .unwrap_i32() for example.
pub enum OutputVec {
    A(Vec<i32>),
    B(Vec<u32>),
    C(Vec<f32>),
    D(Vec<bool>)
}
impl OutputVec {
    fn unwrap_i32(self) -> Vec<i32> {
        match self {
            OutputVec::A(val) => {
                val
            }
            _ => {
                panic!("value is not a u32!");
            }
        }
    }
    fn unwrap_u32(self) -> Vec<u32> {
        match self {
            OutputVec::B(val) => {
                val
            }
            _ => {
                panic!("value is not a u32!");
            }
        }
    }
    fn unwrap_f32(self) -> Vec<f32> {
        match self {
            OutputVec::C(val) => {
                val
            }
            _ => {
                panic!("value is not a f32!");
            }
        }
    }
    fn unwrap_bool(self) -> Vec<bool> {
        match self {
            OutputVec::D(val) => {
                val
            }
            _ => {
                panic!("value is not a bool!");
            }
        }
    }
}

/// Panic macro for invalid types
macro_rules! panic_invalid_type {
    () => {
        panic!("Unsuported type used, supported types are : i32, u32, f32, bool (because of wgsl)");
    };
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
    pub fn create_buffer(&mut self, name: &str, data_type: &str, size: usize, usage: BufferUsage, is_output: bool) {
        let data_type_stride = match data_type {
            "i32" => {4}
            "u32" => {4}
            "f32" => {4}
            "bool" => {1}
            _ => {
                panic_invalid_type!();
            }
        };
        let byte_size = data_type_stride * size;
        self.buffers.push(Buffer {
            buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: byte_size as u64,
                usage: buffer_usage(usage.clone(), is_output),
                mapped_at_creation: false,
            }),
            name: name.to_owned(),
            size: byte_size as u64,
            type_: data_type.to_owned(),
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
            type_: data_type.to_owned(),
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
        // this whole function is divded into 2 parts : the preprocessing of the shader and bindgroup setup and the boring part of the wgpu api
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
                main_headers += &format!("\tlet index: u32 = global_id.x + global_id.y * 65536u;\nif (index >= {}u) {{return;}}", dispatch_linear_len.unwrap());
                
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
                structs.push(format!("struct reserved{i} {{\n\td: [[stride({})]] array<{}>;\n}};\n",b.type_stride, b.type_));
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
                    cpass.dispatch((val % 65536).try_into().unwrap(), (val as f64 / 65536f64).ceil() as u32, 1);
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
                    match self.buffers[buffer_index].type_.as_str() {
                        "i32" => {
                            result = OutputVec::A(bytemuck::cast_slice::<u8, i32>(&data).to_vec());
                        },
                        "u32" => {
                            result = OutputVec::B(bytemuck::cast_slice::<u8, u32>(&data).to_vec());
                        },
                        "f32" => {
                            result = OutputVec::C(bytemuck::cast_slice::<u8, f32>(&data).to_vec());
                        },
                        "bool" => {
                            result = OutputVec::D(data.iter().map(|&e| e != 0).collect());
                        }
                        _ => {
                            panic_invalid_type!();
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
    use crate::libs::*;
    /// The comparison with wgpu hello-compute : we go 205 significant lines of code to only 34
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
        println!("{:?}", result);
    }

    /// The simplest example.
    pub fn example1() {
        let mut device = Device::new();
        let v1 = vec![1i32, 2, 3, 4, 5, 6];
        device.create_buffer_from("v1", &v1, BufferUsage::ReadOnly, false);
        device.create_buffer("output", "i32", v1.len(), BufferUsage::WriteOnly, true);
        let result = device.execute_shader_code(Dispatch::Linear(v1.len()), r"
        fn main() {
            output[index] = v1[index] * 2;
        }
        ").into_iter().next().unwrap().unwrap_i32();
        println!("{:?}", result);
    }

    /// An example with multiple returned buffer .
    pub fn example2() {
        let mut device = Device::new();
        let v = vec![1u32; 30_000_000];
        let start = Instant::now();
        device.create_buffer_from(
            "buf",
            &v,
            BufferUsage::ReadWrite,
            true
        );
        device.create_buffer(
            "buf2",
            "u32",
            30_000_000,
            BufferUsage::ReadOnly,
            false
        );
        device.create_buffer(
            "indices",
            "u32",
            30_000_000,
            BufferUsage::WriteOnly,
            true
        );
        let mut result = device.execute_shader_code(Dispatch::Linear(v.len()), r"
            fn main() {
                indices[index] = index;
                buf[index] = buf[index] + buf2[index] + index;
            }
        ").into_iter();

        let mut output1 = result.next().unwrap().unwrap_u32();
        let mut output2 = result.next().unwrap().unwrap_u32();
        println!("{:?}", output1[0..10].to_owned());
        println!("{:?}", output2[0..10].to_owned());
    }
}
