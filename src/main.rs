mod cfg;
use std::fs;

use rustpython_ast::Stmt;
fn read_file(path: &str) -> String {
    fs::read_to_string(path).expect("File couldn't read")
}

fn main() -> Result<(), rustpython_parser::ParseError> {
    let python_file_path = "./test-file.py";
    let python_code = read_file(python_file_path);
    // Parse ast of the python code.
    let python_ast: Vec<Stmt> = rustpython_parser::Parse::parse_without_path(&python_code)?;
    
    Ok(())
}
