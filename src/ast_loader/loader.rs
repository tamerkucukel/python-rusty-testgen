use rustpython_ast::{Stmt, StmtFunctionDef};
use rustpython_parser::Parse;
use std::fs;

use super::error::AstLoaderError;

//─────────────────────────────────────────────────────────────────────────────

/// Loads a Python AST from a file and extracts function definitions.
pub fn load_ast_from_file(file_path: &str) -> Result<Vec<StmtFunctionDef>, AstLoaderError> {
    // Read the file content
    let file_content =
        fs::read_to_string(file_path).map_err(|e| AstLoaderError::ReadFile(file_path.into(), e))?;

    // Parse the content into an AST
    let ast = Parse::parse_without_path(&file_content)
        .map_err(|e| AstLoaderError::ParseAst(file_path.into(), e))?;

    // Extract function definitions from the AST
    match get_function_defs(ast) {
        Some(function_defs) => Ok(function_defs),
        None => Err(AstLoaderError::ExtractFunctionDefs(file_path.into())),
    }
}

/// Extracts function definitions from a vector of AST statements.
fn get_function_defs(ast: Vec<Stmt>) -> Option<Vec<StmtFunctionDef>> {
    let mut function_defs = Vec::new();
    for stmt in ast {
        if let Stmt::FunctionDef(func_def) = stmt {
            function_defs.push(func_def);
        }
    }

    if function_defs.is_empty() {
        None
    } else {
        Some(function_defs)
    }
}
