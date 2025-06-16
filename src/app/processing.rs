//! Handles the core logic of processing Python functions for test generation.
//!
//! This module includes loading Abstract Syntax Trees (ASTs) from files,
//! building Control Flow Graphs (CFGs), analyzing paths using Z3,
//! and generating Pytest test suite components.

use super::error::AppError;
use super::{verbose_eprintln, verbose_println}; // Macros for conditional logging.
use crate::ast_loader;
use crate::cfg::ControlFlowGraph;
use crate::path::{self, PathConstraintResult, PathScraper};
use crate::testgen::PytestGenerator;
use rustpython_parser::ast::StmtFunctionDef;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

/// Loads Python function definitions (AST nodes) from a specified file.
///
/// # Arguments
/// * `python_file_path` - Path to the Python source file.
/// * `quiet_mode` - Suppresses verbose logging if true.
///
/// # Returns
/// A `Result` containing a vector of `StmtFunctionDef` on success,
/// or an `AppError` on failure (e.g., invalid path, parsing error).
pub fn load_functions_from_file(
    python_file_path: &PathBuf,
    quiet_mode: bool,
) -> Result<Vec<StmtFunctionDef>, AppError> {
    verbose_println!(quiet_mode, "\n[STEP 1] Loading AST from file...");
    let python_file_str = python_file_path
        .to_str()
        .ok_or_else(|| AppError::InvalidPath(python_file_path.display().to_string()))?;

    let ast_func_defs = ast_loader::load_ast_from_file(python_file_str)?;

    if ast_func_defs.is_empty() {
        verbose_println!(quiet_mode, "   => No function definitions found.");
    } else {
        verbose_println!(
            quiet_mode,
            "   => Found {} function definition(s).",
            ast_func_defs.len()
        );
    }
    Ok(ast_func_defs)
}

/// Processes a single Python function definition to generate test components.
///
/// This involves:
/// 1. Building a Control Flow Graph (CFG).
/// 2. Logging CFG details (if not in quiet mode).
/// 3. Scraping execution paths from the CFG.
/// 4. Analyzing paths with Z3 for satisfiability and models.
/// 5. Generating Pytest import statements and test function strings.
///
/// # Arguments
/// * `func_def` - A reference to the AST node of the function definition.
/// * `python_module_name` - The name of the Python module containing the function.
/// * `quiet_mode` - Suppresses verbose logging if true.
/// * `cfg_log_writer` - A mutable reference to a `BufWriter` for logging CFG details.
///
/// # Returns
/// A `Result` containing a tuple:
///   - `bool`: `true` if any concrete test (not just comments) was generated for this function.
///   - `Vec<String>`: A list of required import strings for the generated tests.
///   - `Vec<String>`: A list of generated Pytest function strings.
/// Or an `AppError` if a non-recoverable error occurs during processing.
pub fn process_single_function(
    func_def: &StmtFunctionDef,
    python_module_name: &str,
    quiet_mode: bool,
    cfg_log_writer: &mut BufWriter<File>,
) -> Result<(bool, Vec<String>, Vec<String>), AppError> {
    let func_name = func_def.name.to_string();

    // Print function processing header
    verbose_println!(
        quiet_mode,
        "\n------------------------------------------------------------"
    );
    verbose_println!(quiet_mode, "Function: {}", func_name);
    verbose_println!(
        quiet_mode,
        "------------------------------------------------------------"
    );

    let mut tests_generated_for_this_function = false;
    let mut generated_imports = Vec::new();
    let mut generated_test_functions = Vec::new();

    // Step 2: Build Control Flow Graph (CFG)
    verbose_println!(quiet_mode, "[STEP 2] Building Control Flow Graph (CFG)...");
    let mut cfg_data = ControlFlowGraph::new();
    cfg_data.from_ast(func_def.clone()); // Clone is necessary as from_ast takes ownership.

    // Log CFG details if not in quiet mode
    if !quiet_mode {
        verbose_println!(
            quiet_mode,
            "   => Logging CFG details to cfg_details.log..."
        );
        if let Err(e) = PathScraper::print_paths_to_writer(&func_name, &cfg_data, cfg_log_writer) {
            verbose_eprintln!(
                quiet_mode,
                "   [ERROR] Failed to write CFG details for {}: {}",
                func_name,
                e
            );
            // Continue processing even if CFG logging fails, as it's non-critical for test generation.
        }
    }

    // Step 3: Scraping paths from CFG
    verbose_println!(quiet_mode, "[STEP 3] Scraping paths from CFG...");
    if let Some(paths) = path::PathScraper::get_paths(&cfg_data) {
        if paths.is_empty() {
            verbose_println!(quiet_mode, "   => No executable paths found.");
            return Ok((
                tests_generated_for_this_function,
                generated_imports,
                generated_test_functions,
            ));
        }
        verbose_println!(quiet_mode, "   => Found {} paths.", paths.len());

        // Step 4: Analyzing paths and generating Z3 constraints
        verbose_println!(
            quiet_mode,
            "[STEP 4] Analyzing paths and generating Z3 constraints..."
        );
        let path_constraint_results = path::analyze_paths(&paths, &cfg_data);
        print_constraint_summary(&path_constraint_results, quiet_mode);

        // Step 5: Generating Pytest components
        verbose_println!(quiet_mode, "[STEP 5] Generating Pytest components...");
        let suite = PytestGenerator::generate_suite_for_function(
            &func_name,
            &path_constraint_results,
            &paths,
            &cfg_data,
            Some(python_module_name),
        );
        generated_imports.extend(suite.imports);
        generated_test_functions.extend(suite.test_functions);

        // Check if any actual test definitions were generated (not just comments)
        if generated_test_functions
            .iter()
            .any(|s| s.trim().starts_with("def test_"))
        {
            tests_generated_for_this_function = true;
            verbose_println!(quiet_mode, "   => Generated test components.");
        } else {
            verbose_println!(
                quiet_mode,
                "   => No satisfiable paths led to test generation."
            );
        }
    } else {
        verbose_println!(quiet_mode, "   => No paths found by PathScraper.");
    }

    Ok((
        tests_generated_for_this_function,
        generated_imports,
        generated_test_functions,
    ))
}

/// Prints a summary of Z3 constraint analysis results for each path.
/// This function is only active if `quiet_mode` is false.
///
/// # Arguments
/// * `path_constraint_results` - A slice of `PathConstraintResult` from Z3 analysis.
/// * `quiet_mode` - Suppresses output if true.
fn print_constraint_summary(
    path_constraint_results: &[PathConstraintResult],
    quiet_mode: bool,
) {
    if quiet_mode {
        return;
    }

    verbose_println!(quiet_mode, "   Constraint Analysis Summary:");
    for result in path_constraint_results {
        let status = if result.is_satisfiable {
            "✅ Satisfiable"
        } else {
            "❌ Unsatisfiable"
        };
        verbose_println!(quiet_mode, "     Path {}: {}", result.path_index, status);

        if let Some(model_str) = &result.model {
            verbose_println!(quiet_mode, "       Model:");
            let parsed_model = PytestGenerator::parse_z3_model(model_str);
            if parsed_model.is_empty() {
                if !model_str.trim().is_empty() {
                    verbose_println!(quiet_mode, "         (raw/unparsed) {}", model_str.trim());
                } else {
                    verbose_println!(quiet_mode, "         (empty model string)");
                }
            } else {
                for (var, val) in parsed_model {
                    verbose_println!(quiet_mode, "         {} = {}", var, val);
                }
            }
        }
        if let Some(error) = &result.error {
            verbose_println!(quiet_mode, "       Error: {}", error);
        }
    }
}
