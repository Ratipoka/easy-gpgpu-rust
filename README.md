# Easy GPGPU
A high level, easy to use gpgpu crate based on [`wgpu`](https://github.com/gfx-rs/wgpu).
It is made for very large computations on powerful gpus

## Main goals :

- make general purpose computing very simple
- make it as easy as possible to write wgsl shaders
- deal with binding buffers automatically

## Limitations :

- only types available for buffers : bool, i32, u32, f32
- max buffer byte_size : around 134_000_000 (~33 million i32)
  use device.apply_on_vector to be able to go up to one billion bytes (260 million i32s)
- takes time to initiate the device for the first time (due to wgpu backends)

## Example 

recreating [`wgpu's hello-compute`](https://github.com/gfx-rs/wgpu/tree/v0.12/wgpu/examples/hello-compute) (205 sloc when writen with wgpu)

```rust
use easy_gpgpu::*;
fn wgpu_hello_compute() {
    let mut device = Device::new();
    let v = vec![1u32, 4, 3, 295];
    let result = device.apply_on_vector(v.clone(), r"
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
    collatz_iterations(element)
    ");
    assert_eq!(result, vec![0, 2, 7, 55]);
}
```
=> No binding, no annoying global_id, no need to use a low level api.

You just need to write the minimum amount of wgsl shader code.

## Simplest usage with .apply_on_vector

```rust
use easy_gpgpu::*;
// create a device
let mut device = Device::new();
// create the vector we want to apply a computation on
let v1 = vec![1.0f32, 2.0, 3.0];
// the next line reads : for every element in v1, perform : element = element * 2.0
let v1 = device.apply_on_vector(v1, "element * 2.0");
println!("{v1:?}");
```

## Usage with .execute_shader_code

```rust
//First create a device :
use easy_gpgpu::*;
let mut device = Device::new();
// Then create some buffers, specify if you want to get their content after the execution :
let v1 = vec![1i32, 2, 3, 4, 5, 6];
// from a vector
device.create_buffer_from("v1", &v1, BufferUsage::ReadOnly, false);
// creates an empty buffer
device.create_buffer("output", BufferType::I32, v1.len(), BufferUsage::WriteOnly, true);
// Finaly, execute a shader :
let result = device.execute_shader_code(Dispatch::Linear(v1.len()), r"
fn main() {
    output[index] = v1[index] * 2;
}").into_iter().next().unwrap().unwrap_i32();
assert_eq!(result, vec![2i32, 4, 6, 8, 10, 12])
```
The buffers are available in the shader with the name provided when created with the device.

The `index` variable is provided thanks to the use of `Dispatch::Linear` (index is a u32).

We had only specified one buffer with `is_output: true` so we get only one vector as an output.

We just need to unwrap the data as a vector of i32s with `.unwrap_i32()`

## More Examples

The simplest method to use
```rust
pub fn simplest_apply() {
    let mut device = Device::new();
    let v1 = vec![1.0f32, 2.0, 3.0];
    // the next line reads : for every element in v1, perform : element = element * 2.0
    let v1 = device.apply_on_vector(v1, "element * 2.0");
    println!("{v1:?}");
}
```

An example of device.apply_on_vector with a previously created buffer

```rust
pub fn apply_with_buf() {
    let mut device = Device::new();
    let v1 = vec![2.0f32, 3.0, 5.0, 7.0, 11.0];
    let exponent = vec![3.0];
    device.create_buffer_from("exponent", &exponent, BufferUsage::ReadOnly, false);
    let cubes = device.apply_on_vector(v1, "pow(element, exponent[0u])");
    println!("{cubes:?}")
}
```

The simplest example with device.execute_shader_code : multiplying by 2 every element of a vector.

```rust
pub fn simplest_execute_shader() {
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
```

An example with multiple returned buffer with device.execute_shader_code

```rust
pub fn multiple_returned_buffers() {
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

    let sum = result.next().unwrap().unwrap_u32(); // first returned buffer
    let product = result.next().unwrap().unwrap_u32(); //second returned buffer
    println!("{:?}", sum);
    println!("{:?}", product);
}
```

An example with a custom dispatch that gives access to the global_id variable.

```rust
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
```

An example with a complete pipeline which as you can see, is quite annoying just to multiply a vector by 2.

```rust
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
```

An example where we execute two shaders on the same device, with the same buffers.

```rust
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
```

Even more examples are available in the docs, in the `examples` module

## Link to helpful doc for writing wgsl shaders

The helpful doc : [`wgsl`](https://www.w3.org/TR/WGSL)

WARNING : the wgsl language described in this documentation is not exactly the one used by this crate :

(the wgsl language used in this crate is the same as one used in the wgpu crate)

-> The attributes in the doc are specified with `@attribute` while in this crate there are with `[[attribute]]`
so `@group(0) @binding(0)` becomes `[[group(0), binding(0)]]`

There are also some other minor differences but this doc is very useful for all the [`builtin functions`](https://www.w3.org/TR/WGSL/#builtin-functions)