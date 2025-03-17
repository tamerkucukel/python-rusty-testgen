mod ast_analyzer;
use ast_analyzer::ASTAnalyzer;
use std::fs;

fn read_file(path: &str) -> String {
    fs::read_to_string(path).expect("File couldn't read")
}

fn main() {
    let python_file_path = "./test-file.py";
    let python_code = read_file(python_file_path);
    let mut analyzer = ASTAnalyzer::new();
    let ast = analyzer.generate_ast(&python_code);
    analyzer.visit(&ast);
    let created_test = ASTAnalyzer::generate_test(&analyzer.function_metrics[0]);
    println!("{}", created_test)
}
