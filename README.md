# Easy GPGPU
A high level, easy to use async gpgpu crate based on [`wgpu`](https://github.com/gfx-rs/wgpu).
It is made for very large computations on powerful gpus

Main goals :

- make general purpose computing very simple
- make it as easy as possible to write wgsl shaders
- deal with binding buffers automatically

Limitations :

- only types available for buffers : bool, i32, u32, f32
- max buffer byte_size : around 134_000_000 (~33 million i32)
- depending on the driver, a process will be killed if it takes more than 3 seconds on the gpu
- takes a bit of time to initiate the device (due to wgpu backends)

## Example 

recreating [`wgpu's hello-compute`](https://github.com/gfx-rs/wgpu/tree/v0.12/wgpu/examples/hello-compute) (205 sloc when writen with wgpu)

```
use easy_gpgpu::*;
fn wgpu_hello_compute() {
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
    }"
    ).into_iter().next().unwrap().unwrap_u32();
    assert_eq!(result, vec![0, 2, 7, 55]);
}
```
=> No binding, no annoying global_id, no need to use a low level api.
You just declare the name of the buffer and it is immediately available in the wgsl shader.

## Usage 
```
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

# More Examples

An example with multiple returned buffer
```rust
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
```

An Example with a custom dispatch that gives access to the global_id variable.

```rust
pub fn example_global_id() {
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

An example where we execute two shaders on the same device, with the same buffers.

```rust
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
```

An example of using a very big buffer :

```rust
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
```