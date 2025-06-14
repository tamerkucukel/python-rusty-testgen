use crate::ast_loader;
use crate::cfg::ControlFlowGraph;
use crate::path::{self, PathConstraintResult, PathScraper};
use crate::testgen::PytestGenerator;
use rustpython_parser::ast::StmtFunctionDef;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use super::error::AppError;
use super::{verbose_eprintln, verbose_println};


pub fn load_functions_from_file(
    python_file_path: &PathBuf,
    quiet_mode: bool,
) -> Result<Vec<StmtFunctionDef>, AppError> {
    verbose_println!(quiet_mode, "\n[STEP 1] Loading AST from file..."); // Added newline for spacing
    let ast_func_defs = ast_loader::load_ast_from_file(
        python_file_path
            .to_str()
            .ok_or_else(|| AppError::InvalidPath(python_file_path.display().to_string()))?,
    )?;

    if ast_func_defs.is_empty() {
        // File path context is already provided by the orchestrator's initial message
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

/// Processes a single Python function to generate test components.
/// Returns a tuple: (bool: if any test was generated, Vec<String>: imports, Vec<String>: test function strings).
pub fn process_single_function(
    func_def: &StmtFunctionDef, // This is &Located<StmtFunctionDef_>
    python_module_name: &str,
    quiet_mode: bool,
    cfg_log_writer: &mut BufWriter<File>,
) -> Result<(bool, Vec<String>, Vec<String>), AppError> {
    let func_name = func_def.name.to_string();

    verbose_println!(quiet_mode, "\n------------------------------------------------------------");
    verbose_println!(quiet_mode, "Function: {}", func_name);
    verbose_println!(quiet_mode, "------------------------------------------------------------");

    let mut tests_generated_for_this_function = false;

    // Function name context is now in the header for this section
    verbose_println!(quiet_mode, "[STEP 2] Building Control Flow Graph (CFG)...");
    let mut cfg_data = ControlFlowGraph::new();
    cfg_data.from_ast(func_def.clone()); // Clone is necessary as from_ast takes ownership

    if !quiet_mode {
        verbose_println!(
            quiet_mode,
            "   => Logging CFG details to cfg_details.log..."
        );
        if let Err(e) = PathScraper::print_paths_to_writer(&cfg_data, cfg_log_writer) {
            verbose_eprintln!(
                quiet_mode,
                "   [ERROR] Failed to write CFG details for {}: {}", // func_name is now correct
                func_name,
                e
            );
        }
    }

    verbose_println!(quiet_mode, "[STEP 3] Scraping paths from CFG...");

    let mut generated_imports = Vec::new();
    let mut generated_test_functions = Vec::new();

    if let Some(paths) = path::PathScraper::get_paths(&cfg_data) {
        if paths.is_empty() {
            verbose_println!(quiet_mode, "   => No executable paths found.");
            return Ok((false, generated_imports, generated_test_functions));
        }
        verbose_println!(quiet_mode, "   => Found {} paths.", paths.len());

        verbose_println!(quiet_mode, "[STEP 4] Analyzing paths and generating Z3 constraints...");
        let path_constraint_results = path::analyze_paths(&paths, &cfg_data);

        print_constraint_summary(&path_constraint_results, quiet_mode); // Removed func_name, context is implicit

        verbose_println!(quiet_mode, "[STEP 5] Generating Pytest components...");
        let suite = PytestGenerator::generate_suite_for_function(
            &func_name, // func_name is now correct
            &path_constraint_results,
            &paths,
            &cfg_data,
            Some(python_module_name),
        );
        generated_imports.extend(suite.imports);
        generated_test_functions.extend(suite.test_functions);

        if generated_test_functions.iter().any(|s| s.trim().starts_with("def test_")) {
            tests_generated_for_this_function = true;
            verbose_println!(quiet_mode, "   => Generated test components.");
        } else {
            verbose_println!(quiet_mode, "   => No satisfiable paths led to test generation.");
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

fn print_constraint_summary(
    path_constraint_results: &[PathConstraintResult],
    quiet_mode: bool, // Removed func_name parameter
) {
    if quiet_mode {
        return;
    }
    // func_name context is implicit from the function processing block
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
            if parsed_model.is_empty() && !model_str.trim().is_empty() {
                verbose_println!(quiet_mode, "         (raw) {}", model_str.trim());
            } else if parsed_model.is_empty() {
                verbose_println!(quiet_mode, "         (empty or unparsed model)");
            }
            for (var, val) in parsed_model {
                verbose_println!(quiet_mode, "         {} = {}", var, val);
            }
        }
        if let Some(error) = &result.error {
            verbose_println!(quiet_mode, "       Error: {}", error);
        }
    }
}