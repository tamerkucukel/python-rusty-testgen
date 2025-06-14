use rustpython_ast::{Stmt, StmtFunctionDef};
use rustpython_parser::Parse;
use std::fs;

use super::error::AstLoaderError;

//─────────────────────────────────────────────────────────────────────────────

/// Loads a Python AST from a file and extracts function definitions.
pub fn load_ast_from_file(file_path: &str) -> Result<Vec<StmtFunctionDef>, AstLoaderError> {
    // Read the file content
    // Refactored to use '?' for cleaner error propagation.
    let file_content =
        fs::read_to_string(file_path).map_err(|e| AstLoaderError::ReadFile(file_path.into(), e))?;

    // Parse the content into an AST
    // Refactored to use '?' for cleaner error propagation.
    let ast = Parse::parse_without_path(&file_content)
        .map_err(|e| AstLoaderError::ParseAst(file_path.into(), e))?;

    // Extract function definitions from the AST
    // Refactored to use a more direct error return if no functions are found.
    get_function_defs(ast)
        .ok_or_else(|| AstLoaderError::NoFunctionsFound(file_path.into()))
}

/// Extracts function definitions from a vector of AST statements.
/// Returns `None` if no function definitions are found.
fn get_function_defs(ast: Vec<Stmt>) -> Option<Vec<StmtFunctionDef>> {
    // Replaced explicit for loop with iterator chain for conciseness.
    let function_defs: Vec<StmtFunctionDef> = ast
        .into_iter() // Consumes ast, which is fine as it's not used afterwards.
        .filter_map(|stmt| {
            if let Stmt::FunctionDef(func_def) = stmt {
                Some(func_def)
            } else {
                None
            }
        })
        .collect();

    if function_defs.is_empty() {
        None
    } else {
        Some(function_defs)
    }
}
