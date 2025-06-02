// error module
mod error;
// explorer module
mod explorer;

// constraint module
pub mod constraint;

//─────────────────────────────────────────────────────────────────────────────
// Public re-exports from the explorer module.
//─────────────────────────────────────────────────────────────────────────────
pub use constraint::{analyze_paths, print_paths, PathConstraintResult};
pub use error::{PathError, Z3Error};
pub use explorer::PathScraper;
