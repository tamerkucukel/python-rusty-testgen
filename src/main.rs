use rustpython_parser::{self, ast::*, Parse};
use std::{fs::File, io::Read, path::Path};
fn main() {
    // Specify the path to your Python file
    let python_file_path: &str = "./test-file.py";

    // Read the file content
    let mut file: File = File::open(Path::new(python_file_path)).expect("Could not open file");
    let mut source_code: String = String::new();
    file.read_to_string(&mut source_code)
        .expect("Could not read file");

    // Parse the Python code into an AST
    let generated_ast = match Suite::parse(&source_code, &python_file_path) {
        Ok(ast) => ast,
        Err(err) => panic!("Failed to parse Python file: {err:?}"),
    };

    //Print AST in correct format.
    println!("Generated ast is: {:#?}", generated_ast);

    //Match with statement types.
    for statement in generated_ast.iter() {
        match statement {
            Stmt::FunctionDef(StmtFunctionDef {
                name,
                ..
            }) => {
                println!("Function name: {}", name);
            }
            _ => {}
        }
    }
}
