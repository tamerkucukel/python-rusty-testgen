//! Main application orchestrator.
//!
//! Coordinates the entire test generation process:
//! 1. Initializes logging.
//! 2. Validates input Python file and extracts module name.
//! 3. Loads Python function definitions (ASTs).
//! 4. Initializes a writer for CFG (Control Flow Graph) details.
//! 5. Iterates through each function, delegating processing to `processing::process_single_function`.
//!    This includes CFG building, path analysis with Z3, and Pytest component generation.
//!    The main verbose log (`testgen.log`) is flushed after each function if not in quiet mode.
//! 6. Aggregates generated imports and test functions.
//! 7. Writes the final consolidated Pytest suite to an output file if tests were generated.
//! 8. Provides summary messages to the user.
//!
//! Adheres to command-line arguments like `quiet_mode` for controlling verbosity.

use super::cli::Cli;
use super::error::AppError;
use super::file_handler;
use super::logger;
use super::processing;
use super::{verbose_eprintln, verbose_println}; // Macros for conditional logging.
use std::collections::HashSet;
use std::io::Write; // For BufWriter::flush
use std::path::Path;

/// Runs the main application logic based on parsed command-line arguments.
///
/// # Arguments
/// * `cli` - The `Cli` struct containing parsed command-line arguments.
///
/// # Errors
/// Returns `AppError` if any unrecoverable error occurs during the process,
/// such as critical I/O failures or fatal errors during core processing stages.
///
/// # Performance & I/O
/// - Logger initialization and CFG log writer initialization involve file I/O.
/// - AST loading involves file I/O and parsing.
/// - The main loop processes functions sequentially. The verbose log (`testgen.log`)
///   is flushed after each function (if not quiet), which increases I/O operations
///   but ensures log persistence.
/// - Final test suite content is built in a `String` buffer. `String::with_capacity`
///   is used to pre-allocate some memory, potentially reducing reallocations.
/// - Writing the final suite involves a single buffered write operation, followed by a flush.
pub fn run_app(cli: Cli) -> Result<(), AppError> {
    let python_file_path = &cli.python_file;
    let quiet_mode = cli.quiet;

    // Initialize global logger if not in quiet mode.
    // This setup is done once.
    if !quiet_mode {
        if let Err(e) = logger::init_global_logger("testgen.log") {
            // If logger init fails, print to stderr directly. The application attempts to
            // continue, but verbose file logging will be unavailable. This matches original behavior.
            eprintln!(
                "Warning: Failed to initialize verbose logger (testgen.log): {}. Verbose file logging will be unavailable.",
                e
            );
        } else {
            // This message goes to the newly initialized log file.
            verbose_println!(quiet_mode, "Verbose logging initialized to testgen.log");
            // Initial flush after initialization message.
            if let Err(e) = logger::flush_global_logger() {
                verbose_eprintln!(
                    quiet_mode,
                    "[WARNING] Failed to flush testgen.log after initialization: {}",
                    e
                );
            }
        }
    }

    // Validate Python file and get module name. This is an early check.
    let python_module_name =
        file_handler::validate_python_file_and_get_module(python_file_path, quiet_mode)?;

    verbose_println!(
        quiet_mode,
        "\n============================================================"
    );
    verbose_println!(
        quiet_mode,
        "Processing File: {}",
        python_file_path.display()
    );
    verbose_println!(
        quiet_mode,
        "============================================================"
    );

    // Load all function definitions (ASTs) from the Python file.
    let ast_func_defs = processing::load_functions_from_file(python_file_path, quiet_mode)?;

    if ast_func_defs.is_empty() {
        // If no functions, nothing to do. Message already logged by `load_functions_from_file`.
        if quiet_mode {
            // Provide minimal output for quiet mode.
            println!(
                "Done. No functions found in {}.",
                python_file_path.display()
            );
        }
        // Ensure final flush of testgen.log if it was initialized
        if !quiet_mode {
            if let Err(e) = logger::flush_global_logger() {
                eprintln!(
                    "[WARNING] Failed to perform final flush of testgen.log: {}",
                    e
                );
            }
        }
        return Ok(());
    }

    // Initialize collections for aggregated test suite components.
    // `HashSet` for imports ensures uniqueness.
    let mut all_pytest_imports: HashSet<String> = HashSet::new();
    let mut all_pytest_functions: Vec<String> = Vec::new();
    let mut any_tests_generated_overall = false;

    // Initialize CFG log writer. This writer is passed to each function processing call.
    // The `BufWriter` will manage buffering and flush on drop at the end of `run_app`.
    let cfg_log_file_path = Path::new("cfg_details.log");
    let mut cfg_log_writer =
        file_handler::init_cfg_log_writer(cfg_log_file_path).map_err(|e| {
            verbose_eprintln!(
                quiet_mode,
                "[ERROR] Failed to open CFG details log (cfg_details.log): {}. CFG details will not be logged.",
                e
            );
            AppError::Io(e)
        })?;

    // Process each function definition sequentially.
    for func_def in &ast_func_defs {
        match processing::process_single_function(
            func_def,
            &python_module_name,
            quiet_mode,
            &mut cfg_log_writer, // Pass the shared writer for CFG details.
        ) {
            Ok((generated_tests_for_func, imports, test_funcs)) => {
                if generated_tests_for_func {
                    any_tests_generated_overall = true;
                }
                all_pytest_imports.extend(imports);
                all_pytest_functions.extend(test_funcs);
            }
            Err(e) => {
                verbose_eprintln!(
                    quiet_mode,
                    "[ERROR] During processing of function '{}': {}",
                    func_def.name,
                    e
                );
            }
        }
        // Flush the global verbose logger (testgen.log) after each function if not in quiet mode.
        if !quiet_mode {
            if let Err(e) = logger::flush_global_logger() {
                // Log to stderr as testgen.log itself might be the one failing.
                eprintln!(
                    "[WARNING] Failed to flush testgen.log after processing function '{}': {}",
                    func_def.name, e
                );
            }
        }
    }

    // Explicitly flush the CFG log writer after all functions are processed.
    if let Err(e) = cfg_log_writer.flush() {
        verbose_eprintln!(
            quiet_mode,
            "[WARNING] Failed to flush CFG details log (cfg_details.log): {}. Some CFG data might be lost.",
            e
        );
    }

    // Log completion of function processing stage.
    if !ast_func_defs.is_empty() && !quiet_mode {
        verbose_println!(
            quiet_mode,
            "\n------------------------------------------------------------"
        );
        verbose_println!(quiet_mode, "Function Processing Complete");
        verbose_println!(
            quiet_mode,
            "------------------------------------------------------------"
        );
        // Final flush for messages after loop, before writing output file.
        if let Err(e) = logger::flush_global_logger() {
            eprintln!("[WARNING] Failed to flush testgen.log after function processing complete message: {}", e);
        }
    }

    // Determine if any actual tests (not just comments/placeholders) were generated.
    let should_write_output_file = any_tests_generated_overall
        || !all_pytest_functions
            .iter()
            .all(|s| s.trim().starts_with('#'));

    if should_write_output_file {
        let mut final_pytest_content = String::with_capacity(
            2048 + all_pytest_functions.iter().map(|s| s.len()).sum::<usize>(),
        );

        final_pytest_content.push_str("import pytest\n");

        let mut sorted_imports: Vec<String> = all_pytest_imports.into_iter().collect();
        sorted_imports.sort_unstable();
        for imp in sorted_imports {
            if imp != "import pytest" {
                final_pytest_content.push_str(&imp);
                final_pytest_content.push('\n');
            }
        }
        final_pytest_content.push_str("\n");

        for func_str in all_pytest_functions {
            final_pytest_content.push_str(&func_str);
            final_pytest_content.push_str("\n\n");
        }

        final_pytest_content.push_str("if __name__ == \"__main__\":\n");
        final_pytest_content.push_str("    pytest.main()\n");

        let pytest_output_path = Path::new("test_generated_suite.py");
        match file_handler::write_content_to_file(pytest_output_path, &final_pytest_content) {
            Ok(_) => {
                verbose_println!(
                    quiet_mode,
                    "\n[INFO] All tests written to {}",
                    pytest_output_path.display()
                );
            }
            Err(e) => {
                verbose_eprintln!(
                    quiet_mode,
                    "[ERROR] Failed to write consolidated pytest file ({}): {}",
                    pytest_output_path.display(),
                    e
                );
                // Ensure final flush of testgen.log before returning error
                if !quiet_mode {
                    if let Err(flush_err) = logger::flush_global_logger() {
                        eprintln!(
                            "[WARNING] Failed to perform final flush of testgen.log on error: {}",
                            flush_err
                        );
                    }
                }
                return Err(AppError::Io(e));
            }
        }
    } else if !ast_func_defs.is_empty() {
        verbose_println!(
            quiet_mode,
            "\n[INFO] No tests were generated for any function in {}.",
            python_file_path.display()
        );
    }

    // Final flush of testgen.log before exiting successfully.
    if !quiet_mode {
        if let Err(e) = logger::flush_global_logger() {
            eprintln!(
                "[WARNING] Failed to perform final flush of testgen.log: {}",
                e
            );
        }
    }

    if quiet_mode {
        println!("Done.");
    } else {
        println!(
            "\nTest generation process finished. See 'testgen.log' for verbose output and 'cfg_details.log' for CFG information."
        );
    }

    Ok(())
}
