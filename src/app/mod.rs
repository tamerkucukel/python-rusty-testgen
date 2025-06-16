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
// `super::logger` is correct here because when the macro is expanded in a sibling
// module (e.g., app::orchestrator), `super` refers to the `app` module,
// and `logger` is a submodule of `app`.
macro_rules! verbose_println {
    ($quiet:expr, $($arg:tt)*) => {
        if !$quiet {
            // Use format_args! to create std::fmt::Arguments, avoiding String allocation.
            super::logger::log_verbose_message_args(format_args!($($arg)*));
        }
    };
}

macro_rules! verbose_eprintln {
    ($quiet:expr, $($arg:tt)*) => {
        if !$quiet {
            // Use format_args! to create std::fmt::Arguments, avoiding String allocation.
            super::logger::log_verbose_error_args(format_args!($($arg)*));
        }
    };
}

// These `use` statements bring the macros into scope for all sibling modules
// within the `app` module, allowing them to be called directly.
use verbose_eprintln;
use verbose_println;
