mod ast_analyzer;
use ast_analyzer::ASTAnalyzer;
use rustpython_ast::Stmt;
use rustpython_parser::{ast, Parse};
use std::fs;

fn read_file(path: &str) -> String {
    fs::read_to_string(path).expect("File couldn't read")
}

fn generate_ast(code: &str) -> Vec<Stmt> {
    ast::Suite::parse_without_path(code).expect("Couldn't parse python code")
}

fn main() {
    let python_file_path = "./test-file.py";
    let python_code = read_file(python_file_path);
    let ast = generate_ast(&python_code);
    println!("{:#?}", ast);
    let mut analyzer = ASTAnalyzer::new();
    analyzer.visit(&ast);
}
