use thiserror::Error;

//─────────────────────────────────────────────────────────────────────────────

/// Error type for AST loading operations.
/// This error type is used to represent various errors that can occur
/// while loading and processing Python ASTs.
#[derive(Error, Debug)]
pub enum AstLoaderError {
    /// Error when reading a file.
    #[error("Failed to read file '{0}': {1}")]
    ReadFile(String, std::io::Error),

    /// Error when parsing the AST.
    #[error("Failed to parse AST from '{0}': {1}")]
    ParseAst(String, rustpython_parser::ParseError),

    /// Error when extracting function definitions from the AST.
    #[error("Failed to extract function definitions from AST, you may not have functions ?: {0}")]
    ExtractFunctionDefs(String),

    /// Error when no function definitions are found in the AST.
    #[error("No function definitions found in the AST for file '{0}'")]
    NoFunctionsFound(String),
}
