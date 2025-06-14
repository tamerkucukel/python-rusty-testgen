use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[clap(author, version, about = "Generates pytest tests for Python functions from a given file.", long_about = None)]
pub struct Cli {
    /// Python file to generate tests for
    pub python_file: PathBuf,

    /// Suppress verbose output, only printing 'Done.' on success or errors.
    #[clap(short, long)]
    pub quiet: bool,
}