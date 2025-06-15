mod app;
mod ast_loader;
mod cfg;
mod path;
mod testgen; // This now refers to src/app/mod.rs

use app::{run_app, AppError, Cli}; // Imports from src/app/mod.rs
use clap::Parser;

fn main() -> Result<(), AppError> {
    let cli = Cli::parse();
    run_app(cli)
}
