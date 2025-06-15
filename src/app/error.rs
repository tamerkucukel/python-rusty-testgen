use thiserror::Error;

// Custom Application Error
#[derive(Error, Debug)]
pub enum AppError {
    #[error("File I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("AST loading error: {0}")]
    AstLoad(#[from] crate::ast_loader::error::AstLoaderError),
    #[error("Z3 processing error: {0}")]
    Z3(#[from] crate::path::error::Z3Error),
    #[error("Invalid file path: {0}")]
    InvalidPath(String),
    #[error("General error: {0}")]
    General(String),
}
