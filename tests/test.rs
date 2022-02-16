use easy_gpgpu::{Device, BufferUsage, Dispatch, BufferType, Command, remove_comments};
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

    fn number_of_seven_in_digit_product(number: u32) -> u32 {
        let mut p: u32 = 1;
        let mut n: u32 = number;
        loop {
            if n == 0 {break;}
            p = p * (n % 10);
            n = n / 10;
        }
        let mut nb_seven: u32 = 0;
        loop {
            if p == 0 {break;}
            if p % 10 == 7 {
                nb_seven = nb_seven + 1;
            }
            p = p / 10;
        }
        return nb_seven;
    }

    let mut vec_cpu = (0..(size as u32)).collect::<Vec<_>>();
    vec_cpu = vec_cpu.iter_mut().map(|e| {
        number_of_seven_in_digit_product(e.clone())
    }).collect();
    assert_eq!(result, vec_cpu);
}

#[test]
fn test_shader_two_steps() {
    let mut device = Device::new();
    let v = vec![1u32, 2, 3, 4];
    device.create_buffer_from("buf1", &v, BufferUsage::ReadWrite, true);
    let result = device.execute_shader_module(&device.create_shader_module(Dispatch::Linear(v.len()), "
    fn main() {
        buf1[index] = buf1[index] * 17u;
    }
    ")).into_iter().next().unwrap().unwrap_u32();
    assert_eq!(result, vec![17u32, 34, 51, 68]);
}

#[test]
fn custom_pipeline() {
    let mut device = Device::new();
    let v = vec![1u32, 2, 3, 4];
    device.create_buffer_from("buf1", &v, BufferUsage::ReadWrite, false);
    device.create_buffer("out", BufferType::U32, v.len(), BufferUsage::WriteOnly, true);
    let mut commands = vec![];
    let shader_command = device.create_shader_module(Dispatch::Linear(v.len()), "
    fn main() {
        // out[index] = index;
        buf1[index] = buf1[index] * 17u;
    }
    ");
    commands.push(Command::Shader(shader_command));
    commands.push(Command::Copy("buf1", "out"));
    device.execute_commands(commands);
}

#[test]
fn test_remove_comments() {
    let code = "
    fn main () {
        // this is a line comment
        out = index [0 / 0] * something; // an other line comment
        /*this is the weirdest comment *// because this is outside
        buf1[index/34] = 3*buf2[index] + 5*buf3 /*this is a weird comment*/ 3;
    }
    ";
    let tmp_code = remove_comments(code.to_string());
    assert_eq!(tmp_code, "
    fn main () {
        
        out = index [0 / 0] * something; 
        / because this is outside
        buf1[index/34] = 3*buf2[index] + 5*buf3  3;
    }
    ")
}

#[test]
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
    // let times2 = device.apply_on_vector(v, "element * 2u");
    assert_eq!(result, vec![0, 2, 7, 55]);
}

/// Test of the maximum sized apply_on_vector
#[test]
fn large_scale_apply() {
    let mut device = Device::new();
    let size = 260_000_000;
    let v = vec![1u32; size];
    let v = device.apply_on_vector(v, "element * 2u");
    assert_eq!(v, vec![2u32; size]);
}

/// A test of the complete pipeline (which as you can see is really unpractical)
#[test]
fn complete_pipeline() {
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