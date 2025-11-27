use std::fs;
use shaderc::{Compiler, ShaderKind};


pub const COMPILED_SHADER_PATH: &str = "shaders/compiled/";
pub const SHADER_PATH: &str = "shaders/src/";

fn main() {
    let compiler = Compiler::new().unwrap();

    for entry in fs::read_dir(SHADER_PATH).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if let Some(ext) = path.extension() {
            let kind = match ext.to_str().unwrap() {
                "vert" => Some(ShaderKind::Vertex),
                "frag" => Some(ShaderKind::Fragment),
                _ => None,
            };

            if let Some(shader_kind) = kind {
                let src = fs::read_to_string(&path).unwrap();
                let filename = path.file_name().unwrap().to_str().unwrap();
                let binary = compiler.compile_into_spirv(&src, shader_kind, filename, "main", None).unwrap();
                let spv_path = format!("{}{}.spv", COMPILED_SHADER_PATH, filename);
                fs::write(spv_path, binary.as_binary_u8()).unwrap();
            }
        }
    }
}