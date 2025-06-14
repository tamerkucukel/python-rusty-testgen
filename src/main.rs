mod ast_loader;
mod cfg;
mod path;
mod testgen;
mod app; // This now refers to src/app/mod.rs

use app::{run_app, Cli, AppError}; // Imports from src/app/mod.rs
use clap::Parser;

fn main() -> Result<(), AppError> {
    let cli = Cli::parse();
    run_app(cli)
}
