mod ast_loader;
mod cfg;
mod path;
mod testgen;

use std::fs;
use std::path::Path;
use clap::Parser;

/// Helper macro for conditional verbose printing
macro_rules! verbose_println {
    ($quiet:expr, $($arg:tt)*) => {
        if !$quiet {
            println!($($arg)*);
        }
    };
}

/// Helper macro for conditional verbose error printing
macro_rules! verbose_eprintln {
    ($quiet:expr, $($arg:tt)*) => {
        if !$quiet {
            eprintln!($($arg)*);
        }
    };
}

#[derive(Parser, Debug)]
#[clap(author, version, about = "Generates pytest tests for Python functions from a given file.", long_about = None)]
struct Cli {
    /// Python file to generate tests for
    python_file: String,

    /// Suppress verbose output, only printing 'Done.' on success or errors.
    #[clap(short, long)]
    quiet: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let python_file_path_str = &cli.python_file;
    let quiet_mode = cli.quiet;

    let python_file_path = Path::new(python_file_path_str);

    if !python_file_path.exists() {
        verbose_eprintln!(quiet_mode, "Error: File not found: {}", python_file_path_str);
        return Err(format!("File not found: {}", python_file_path_str).into());
    }
    if !python_file_path.is_file() {
        verbose_eprintln!(quiet_mode, "Error: Path is not a file: {}", python_file_path_str);
        return Err(format!("Path is not a file: {}", python_file_path_str).into());
    }

    let python_module_name = python_file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown_module");

    verbose_println!(quiet_mode, "Starting test generation for: {}", python_file_path_str);

    verbose_println!(quiet_mode, "Step 1: Loading AST from file...");
    let ast_func_defs = match ast_loader::load_ast_from_file(python_file_path_str) {
        Ok(defs) => defs,
        Err(e) => {
            verbose_eprintln!(quiet_mode, "Error loading AST: {}. Invalid file structure or Python syntax error.", e);
            return Err(e.into());
        }
    };

    if ast_func_defs.is_empty() {
        verbose_println!(quiet_mode, "No function definitions found in {}.", python_file_path_str);
        if quiet_mode { println!("Done."); }
        return Ok(());
    }
    verbose_println!(quiet_mode, "Found {} function definition(s).", ast_func_defs.len());

    let mut tests_generated_overall = false;

    for func_def in &ast_func_defs {
        let func_name = func_def.name.to_string();
        verbose_println!(quiet_mode, "\nProcessing function: {}", func_name);

        verbose_println!(quiet_mode, "Step 2: Building Control Flow Graph (CFG) for {}...", func_name);
        let mut cfg_data = cfg::ControlFlowGraph::new();
        cfg_data.from_ast(func_def.clone());

        verbose_println!(quiet_mode, "Step 3: Scraping paths from CFG for {}...", func_name);
        if let Some(paths) = path::PathScraper::get_paths(&cfg_data) {
            if paths.is_empty() {
                verbose_println!(quiet_mode, "No executable paths found for function {}.", func_name);
                continue;
            }
            verbose_println!(quiet_mode, "Found {} paths for {}.", paths.len(), func_name);

            verbose_println!(quiet_mode, "Step 4: Analyzing paths and generating Z3 constraints for {}...", func_name);
            let path_constraint_results = path::analyze_paths(&paths, &cfg_data);

            // Optional: Print detailed constraint analysis results (if not in quiet mode)
            if !quiet_mode {
                verbose_println!(quiet_mode, "\nConstraint Analysis Summary for {}:", func_name);
                for result in &path_constraint_results {
                    if result.is_satisfiable {
                        verbose_println!(quiet_mode, "  Path {}: ✅ Satisfiable. Model: {}", result.path_index, result.model.as_deref().unwrap_or("N/A"));
                    } else {
                        verbose_println!(quiet_mode, "  Path {}: ❌ Unsatisfiable. Error: {}", result.path_index, result.error.as_ref().map_or_else(|| "N/A".to_string(), |e| e.to_string()));
                    }
                }
            }

            verbose_println!(quiet_mode, "Step 5: Generating Pytest code for {}...", func_name);
            let pytest_string = testgen::PytestGenerator::generate_pytest_file_string(
                &func_name,
                &path_constraint_results,
                &paths,
                &cfg_data,
                Some(python_module_name),
            );

            if !pytest_string.contains("def test_") { // A simple check if any test function was actually generated
                verbose_println!(quiet_mode, "No satisfiable paths led to test generation for function {}.", func_name);
            } else {
                let test_file_name = format!("test_{}.py", func_name.to_lowercase().replace(" ", "_"));
                verbose_println!(quiet_mode, "Attempting to write tests to {}...", test_file_name);
                match fs::write(&test_file_name, pytest_string) {
                    Ok(_) => {
                        verbose_println!(quiet_mode, "Successfully generated and wrote tests to {}", test_file_name);
                        tests_generated_overall = true;
                    }
                    Err(e) => {
                        verbose_eprintln!(quiet_mode, "Error: Failed to write test file {}: {}", test_file_name, e);
                        // Continue to next function if one fails to write
                    }
                }
            }
        } else {
            verbose_println!(quiet_mode, "No paths found by PathScraper for function {}.", func_name);
        }
    }

    if !tests_generated_overall && ast_func_defs.is_empty() {
         verbose_println!(quiet_mode, "No tests were generated because no functions were found or processed.");
    } else if !tests_generated_overall {
        verbose_println!(quiet_mode, "No tests were generated for any function (e.g. no satisfiable paths or other issues).");
    }


    if quiet_mode {
        println!("Done.");
    } else {
        println!("\nTest generation process finished.");
    }

    Ok(())
}
