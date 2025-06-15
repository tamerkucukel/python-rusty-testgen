use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Error as IoError, Write};
use std::path::{Path, PathBuf};
// Use super:: for macros defined in app/mod.rs
use super::error::AppError;
use super::verbose_eprintln; // These macros now write to the log file

pub fn validate_python_file_and_get_module(
    python_file_path: &PathBuf,
    quiet_mode: bool, // quiet_mode is passed to macros
) -> Result<String, AppError> {
    if !python_file_path.exists() {
        // This message will go to the log file if not quiet, and also to stderr via AppError
        verbose_eprintln!(
            quiet_mode,
            "Input Error: File not found: {}",
            python_file_path.display()
        );
        return Err(AppError::General(format!(
            "File not found: {}",
            python_file_path.display()
        )));
    }
    if !python_file_path.is_file() {
        verbose_eprintln!(
            quiet_mode,
            "Input Error: Path is not a file: {}",
            python_file_path.display()
        );
        return Err(AppError::General(format!(
            "Path is not a file: {}",
            python_file_path.display()
        )));
    }

    let python_module_name = python_file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown_module")
        .to_string();

    Ok(python_module_name)
}

/// Writes content to a specified file, overwriting it if it exists.
pub fn write_content_to_file(file_path: &Path, content: &str) -> Result<(), IoError> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(file_path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(content.as_bytes())?;
    writer.flush()?; // Ensure all content is written
    Ok(())
}

/// Initializes a BufWriter for the CFG details log file.
pub fn init_cfg_log_writer(file_path: &Path) -> Result<BufWriter<File>, IoError> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true) // Overwrite CFG log each run
        .open(file_path)?;
    Ok(BufWriter::new(file))
}
