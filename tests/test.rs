use easy_gpgpu::{Device, BufferUsage, Dispatch};
#[test]
fn example2() {
    let mut device = Device::new();
    let v = vec![1u32; 30_000_000];
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
        BufferUsage::ReadWrite,
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
            buf2[index] = index * 3u;
            buf[index] = buf[index] + buf2[index];
        }
    ").into_iter();
    let output1 = result.next().unwrap().unwrap_u32();
    let output2 = result.next().unwrap().unwrap_u32();
    println!("{:?}", output1[0..10].to_owned());
    println!("{:?}", output2[0..10].to_owned());
    assert_eq!(output1, (1..90_000_000).step_by(3).collect::<Vec<_>>());
    assert_eq!(output2, (0..30_000_000).collect::<Vec<_>>());
}


// #[test]
fn reusing_device() {
    // example 1
    let mut device = Device::new();
    let v1 = vec![1i32, 2, 3, 4, 5, 6];
    device.create_buffer_from("v1", &v1, BufferUsage::ReadOnly, false);
    device.create_buffer("output", "i32", v1.len(), BufferUsage::WriteOnly, true);
    let result = device.execute_shader_code(Dispatch::Linear(v1.len()), r"
    fn main() {
        output[index] = v1[index] * 2;
    }
    ").into_iter().next().unwrap().unwrap_i32();
    assert_eq!(result, vec![2i32, 4, 6, 8, 10, 12]);

    // example 2
    let v = vec![1u32; 30_000_000];
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
    let output1 = result.next().unwrap().unwrap_u32();
    let output2 = result.next().unwrap().unwrap_u32();
    assert_eq!(output2, (0..30_000_000).collect::<Vec<_>>());
    assert_eq!(output1, (1..30_000_001).collect::<Vec<_>>());
}