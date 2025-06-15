use once_cell::sync::Lazy;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Error as IoError, Write};
use std::sync::Mutex;

// Global static logger instance
static LOGGER: Lazy<Mutex<Option<BufWriter<File>>>> = Lazy::new(|| Mutex::new(None));

/// Initializes the global logger to write to the specified file path.
/// If the file doesn't exist, it's created. If it exists, output is appended.
pub fn init_global_logger(log_file_path: &str) -> Result<(), IoError> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(log_file_path)?;
    let writer = BufWriter::new(file);
    let mut logger_guard = LOGGER.lock().expect("Logger mutex poisoned");
    *logger_guard = Some(writer);
    Ok(())
}

/// Writes a verbose message to the global logger.
pub fn log_verbose_message(message: String) {
    if let Ok(mut logger_guard) = LOGGER.lock() {
        if let Some(writer) = logger_guard.as_mut() {
            if writeln!(writer, "{}", message).is_err() {
                // Fallback to stderr if log writing fails
                eprintln!("Fallback (log write failed): {}", message);
            }
            // Optionally flush, though BufWriter handles buffering.
            // writer.flush().ok();
        }
        // If logger is None (not initialized), verbose messages are suppressed.
    } else {
        // Mutex was poisoned, highly unlikely but good to acknowledge.
        eprintln!("Fallback (logger mutex error): {}", message);
    }
}

/// Writes a verbose error message to the global logger.
pub fn log_verbose_error(message: String) {
    if let Ok(mut logger_guard) = LOGGER.lock() {
        if let Some(writer) = logger_guard.as_mut() {
            if writeln!(writer, "ERROR: {}", message).is_err() {
                eprintln!("Fallback (log write failed) ERROR: {}", message);
            }
        } else {
            // If logger is None, verbose errors are also suppressed from the log file.
            // Consider if these should still go to stderr if logger isn't set up.
            // For now, consistent with verbose_println.
        }
    } else {
        eprintln!("Fallback (logger mutex error) ERROR: {}", message);
    }
}
