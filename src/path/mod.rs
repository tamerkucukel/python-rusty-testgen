// error module
pub(crate) mod error;
// explorer module
mod explorer;

// constraint module
pub mod constraint;

//─────────────────────────────────────────────────────────────────────────────
// Public re-exports from the explorer module.
//─────────────────────────────────────────────────────────────────────────────
pub use constraint::{analyze_paths, PathConstraintResult};
pub use explorer::PathScraper;
