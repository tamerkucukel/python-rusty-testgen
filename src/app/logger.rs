//! Handles global application logging to a file.
//!
//! This module provides a simple, globally accessible file logger. It's initialized
//! once at the beginning of the application run (if not in quiet mode). Log messages
//! are written through a `BufWriter` for efficiency. The logger is thread-safe
//! due to the use of a `Mutex`.
//!
//! Logging functions are designed to be called via the `verbose_println!` and
//! `verbose_eprintln!` macros defined in `src/app/mod.rs`, which use `format_args!`
//! to avoid unnecessary string allocations.

use once_cell::sync::Lazy;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Error as IoError, Write}; // Added Write for flush
use std::sync::Mutex;

/// Global static logger instance, wrapped in a `Mutex` for thread-safe access.
///
/// It holds an `Option<BufWriter<File>>` to allow for lazy initialization.
/// `once_cell::sync::Lazy` ensures that the `Mutex` and its `None` content
/// are initialized only once, on first access.
///
/// # Concurrency
/// The `Mutex` ensures that multiple threads attempting to log concurrently will do so
/// sequentially, preventing interleaved log messages or data races on the `BufWriter`.
static LOGGER: Lazy<Mutex<Option<BufWriter<File>>>> = Lazy::new(|| Mutex::new(None));

/// Initializes the global logger to write to the specified file path.
///
/// If the file doesn't exist, it's created. If it exists, its content is truncated
/// before writing new log messages, ensuring a fresh log for each run.
///
/// # Arguments
/// * `log_file_path` - The path to the log file (e.g., "testgen.log").
///
/// # Errors
/// Returns an `IoError` if the file cannot be opened or created.
///
/// # I/O
/// - Uses `OpenOptions` to control file creation and truncation.
/// - Wraps the `File` in a `BufWriter` to buffer writes, improving I/O performance
///   by reducing the number of direct system calls.
pub fn init_global_logger(log_file_path: &str) -> Result<(), IoError> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(log_file_path)?;
    let writer = BufWriter::new(file);

    let mut logger_guard = LOGGER
        .lock()
        .expect("Fatal: Logger mutex has been poisoned. This indicates a prior panic while holding the logger lock.");
    *logger_guard = Some(writer);
    Ok(())
}

/// Writes a verbose message to the global logger using `std::fmt::Arguments`.
///
/// This function is intended to be called by macros like `verbose_println!`
/// which pass `format_args!(...)`. This avoids intermediate `String` allocation
/// for the log message.
///
/// # Arguments
/// * `args` - The `std::fmt::Arguments` to log.
pub fn log_verbose_message_args(args: std::fmt::Arguments) {
    match LOGGER.lock() {
        Ok(mut logger_guard) => {
            if let Some(writer) = logger_guard.as_mut() {
                if writer.write_fmt(format_args!("{}\n", args)).is_err() {
                    eprintln!("Fallback (log write failed): {}", args);
                }
            }
        }
        Err(poison_error) => {
            eprintln!("Fallback (logger mutex error: {}): {}", poison_error, args);
        }
    }
}

/// Writes a verbose error message, prefixed with "ERROR: ", to the global logger
/// using `std::fmt::Arguments`.
///
/// # Arguments
/// * `args` - The `std::fmt::Arguments` for the error message.
pub fn log_verbose_error_args(args: std::fmt::Arguments) {
    match LOGGER.lock() {
        Ok(mut logger_guard) => {
            if let Some(writer) = logger_guard.as_mut() {
                if writer.write_fmt(format_args!("ERROR: {}\n", args)).is_err() {
                    eprintln!("Fallback (log write failed) ERROR: {}", args);
                }
            }
        }
        Err(poison_error) => {
            eprintln!(
                "Fallback (logger mutex error: {}) ERROR: {}",
                poison_error, args
            );
        }
    }
}

/// Flushes the global logger's `BufWriter`.
///
/// This ensures that all buffered log messages are written to the underlying file.
/// It should be called when immediate persistence of logs is required, for example,
/// after processing a significant unit of work.
///
/// # Returns
/// `Ok(())` if flushing was successful or if the logger was not initialized (a no-op).
/// An `IoError` if flushing failed.
///
/// # Errors
/// Returns an `IoError` if the underlying flush operation fails.
/// Returns `Ok(())` if the logger mutex is poisoned, printing a message to stderr,
/// as attempting to flush a poisoned logger is problematic.
pub fn flush_global_logger() -> Result<(), IoError> {
    match LOGGER.lock() {
        Ok(mut logger_guard) => {
            if let Some(writer) = logger_guard.as_mut() {
                writer.flush()?;
            }
            // If logger is None (not initialized), flushing is a no-op.
            Ok(())
        }
        Err(poison_error) => {
            // Mutex was poisoned. Flushing is likely not possible or safe.
            eprintln!(
                "Error: Could not flush logger because mutex was poisoned: {}",
                poison_error
            );
            // Consider what error to return. For now, let's say it's not an IO error
            // in the traditional sense of flushing, but a state error.
            // To keep it simple and consistent with other error handling,
            // we can return Ok and rely on the stderr message.
            // Alternatively, define a custom error or map to a generic IoError.
            // For now, returning Ok to avoid halting if this specific error occurs.
            Ok(())
        }
    }
}
