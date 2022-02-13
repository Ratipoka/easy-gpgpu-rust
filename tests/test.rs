use easy_gpgpu::{Device, BufferUsage, Dispatch, BufferType};
#[test]
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
#[test]
pub fn example2() {
    let mut device = Device::new();
    let v = vec![1u32, 2, 3];
    let v2 = vec![3u32, 4, 5];
    let v3 = vec![7u32, 8, 9];
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
    assert_eq!(sum, vec![11, 14, 17]);
    assert_eq!(product, vec![231, 448, 765]);
}

// An Example with a custom dispatch that gives acess to the global_id variable.
#[test]
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

/// The comparison with wgpu hello-compute : we go from 205 significant lines of code to only 34
#[test]
pub fn example_wgpu_hello_compute() {
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


#[test]
fn reusing_device() {
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