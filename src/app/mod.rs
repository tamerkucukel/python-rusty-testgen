mod cli;
mod error;
mod file_handler; // Keep name file_handler for now
mod logger; // Add logger module
mod orchestrator;
mod processing;

pub use cli::Cli;
pub use error::AppError;
pub use orchestrator::run_app;

// Macros for use by child modules of app (orchestrator, processing, file_handler)
// These macros call functions from the app::logger module.
macro_rules! verbose_println {
    ($quiet:expr, $($arg:tt)*) => {
        if !$quiet {
            super::logger::log_verbose_message(format!($($arg)*));
        }
    };
}

macro_rules! verbose_eprintln {
    ($quiet:expr, $($arg:tt)*) => {
        if !$quiet {
            super::logger::log_verbose_error(format!($($arg)*));
        }
    };
}
use verbose_eprintln;
use verbose_println;
