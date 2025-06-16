//! Provides utility functions for file system operations critical to the application.
//!
//! This includes validating Python file paths, extracting module names,
//! writing content to files, and initializing log writers. It uses macros
//! from the parent `app` module for verbose logging.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Error as IoError, Write};
use std::path::{Path, PathBuf};
// Use super:: for macros defined in app/mod.rs
use super::error::AppError;
use super::verbose_eprintln; // These macros now write to the log file if logger is initialized.

/// Validates the given Python file path and extracts a module name from it.
///
/// Checks if the path exists and points to a file. The module name is derived
/// from the file stem.
///
/// # Arguments
/// * `python_file_path` - A `PathBuf` to the Python file.
/// * `quiet_mode` - A boolean indicating whether to suppress verbose logging.
///
/// # Errors
/// Returns `AppError::General` if the path is invalid (not found or not a file).
pub fn validate_python_file_and_get_module(
    python_file_path: &PathBuf,
    quiet_mode: bool,
) -> Result<String, AppError> {
    if !python_file_path.exists() {
        let error_msg = format!("File not found: {}", python_file_path.display());
        verbose_eprintln!(quiet_mode, "Input Error: {}", error_msg);
        return Err(AppError::General(error_msg));
    }
    if !python_file_path.is_file() {
        let error_msg = format!("Path is not a file: {}", python_file_path.display());
        verbose_eprintln!(quiet_mode, "Input Error: {}", error_msg);
        return Err(AppError::General(error_msg));
    }

    let python_module_name = python_file_path
        .file_stem()
        .and_then(|os_str| os_str.to_str())
        .map(|s| s.to_string()) // Allocate String if valid stem.
        .ok_or_else(|| {
            // This case is unlikely if `is_file` passed and it has an extension,
            // but good to handle robustly.
            let error_msg = format!(
                "Could not determine module name from file: {}",
                python_file_path.display()
            );
            verbose_eprintln!(quiet_mode, "Input Error: {}", error_msg);
            AppError::General(error_msg)
        })?;

    Ok(python_module_name)
}

/// Writes string content to a specified file, creating or overwriting it.
///
/// Ensures the file is created if it doesn't exist and truncated if it does.
/// The entire content is written using a `BufWriter` for efficiency, and the
/// writer is explicitly flushed to ensure data persistence before the function returns.
///
/// # Arguments
/// * `file_path` - The `Path` to the file to write to.
/// * `content` - The string slice (`&str`) to write to the file.
///
/// # Errors
/// Returns an `IoError` if any file operation (opening, writing, flushing) fails.
///
/// # I/O
/// - Uses `OpenOptions` for precise control over file mode.
/// - `BufWriter` is employed to minimize direct system write calls.
/// - `writer.flush()?` is crucial here to ensure that all buffered data is written
///   to the underlying file before the function returns, guaranteeing that the
///   caller sees the complete file content immediately after a successful call.
pub fn write_content_to_file(file_path: &Path, content: &str) -> Result<(), IoError> {
    let file = OpenOptions::new()
        .create(true) // Create if it doesn't exist.
        .write(true) // Open for writing.
        .truncate(true) // Truncate to 0 bytes if it exists.
        .open(file_path)?;
    let mut writer = BufWriter::new(file); // Default buffer capacity.
    writer.write_all(content.as_bytes())?;
    writer.flush()?; // Ensure all buffered content is written to disk.
    Ok(())
}

/// Initializes and returns a `BufWriter<File>` for the CFG details log file.
///
/// The file is created if it doesn't exist and truncated if it does. This ensures
/// that the CFG log is fresh for each application run, containing only the details
/// from the current execution.
///
/// # Arguments
/// * `file_path` - The `Path` to the CFG log file (e.g., "cfg_details.log").
///
/// # Errors
/// Returns an `IoError` if the file cannot be opened or created.
///
/// # I/O
/// - Similar to `write_content_to_file`, uses `OpenOptions` and `BufWriter`.
/// - No explicit `flush` is called here because the `BufWriter` is returned.
///   It will be flushed when its buffer fills or when it is dropped (e.g., when
///   the `cfg_log_writer` in `orchestrator.rs` goes out of scope). This is an
///   appropriate strategy for a log file that might receive multiple writes.
pub fn init_cfg_log_writer(file_path: &Path) -> Result<BufWriter<File>, IoError> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true) // Overwrite CFG log each run.
        .open(file_path)?;
    Ok(BufWriter::new(file)) // Default buffer capacity.
}
