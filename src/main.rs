mod ast_analyzer;
use std::fs;

use ast_analyzer::{analyze_python_code, generate_tests_for_function};

fn read_file(path: &str) -> String {
    fs::read_to_string(path).expect("File couldn't read")
}

fn main() -> std::io::Result<()> {
    let python_file_path = "./test-file.py";
    let python_code = read_file(python_file_path);
    let function_metrics = analyze_python_code(&python_code);
    for metric in function_metrics.unwrap() {
        let test = generate_tests_for_function(&metric);
        fs::write("test_generated.py", test)?;
    }
    Ok(())
}
