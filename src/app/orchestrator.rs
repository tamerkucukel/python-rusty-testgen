use super::cli::Cli;
use super::error::AppError;
use super::file_handler;
use super::logger;
use super::processing;
use super::{verbose_eprintln, verbose_println};
use std::collections::HashSet;
use std::path::Path;

pub fn run_app(cli: Cli) -> Result<(), AppError> {
    let python_file_path = &cli.python_file;
    let quiet_mode = cli.quiet;

    if !quiet_mode {
        if let Err(e) = logger::init_global_logger("testgen.log") {
            eprintln!(
                "Fatal: Failed to initialize verbose logger (testgen.log): {}",
                e
            );
            return Err(AppError::Io(e));
        }
        verbose_println!(quiet_mode, "Verbose logging initialized to testgen.log");
    }

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

    let ast_func_defs = processing::load_functions_from_file(python_file_path, quiet_mode)?;

    if ast_func_defs.is_empty() {
        if quiet_mode {
            println!("Done. No functions found.");
        }
        // The "No function definitions found" message is already logged by load_functions_from_file
        return Ok(());
    }

    let mut all_pytest_imports: HashSet<String> = HashSet::new();
    let mut all_pytest_functions: Vec<String> = Vec::new();
    let mut any_tests_generated_overall = false;

    let cfg_log_file_path = Path::new("cfg_details.log");
    let mut cfg_log_writer = file_handler::init_cfg_log_writer(cfg_log_file_path).map_err(|e| {
        verbose_eprintln!(
            quiet_mode,
            "[ERROR] Failed to open CFG details log (cfg_details.log): {}", // Enhanced error
            e
        );
        AppError::Io(e)
    })?;

    for func_def in &ast_func_defs {
        match processing::process_single_function(
            func_def,
            &python_module_name,
            quiet_mode,
            &mut cfg_log_writer,
        ) {
            Ok((generated_for_func, imports, test_funcs)) => {
                if generated_for_func {
                    any_tests_generated_overall = true;
                }
                all_pytest_imports.extend(imports);
                all_pytest_functions.extend(test_funcs);
            }
            Err(e) => {
                // Keep function name here as it's an error summary for this function
                verbose_eprintln!(
                    quiet_mode,
                    "[ERROR] During processing of function {}: {}",
                    func_def.name,
                    e
                );
            }
        }
    }
    // Add a final separator after all functions are processed
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
    }

    if any_tests_generated_overall
        || !all_pytest_functions
            .iter()
            .all(|s| s.trim().starts_with('#'))
    {
        let mut final_pytest_content = String::new();
        final_pytest_content.push_str("import pytest\n\n");

        let mut sorted_imports: Vec<String> = all_pytest_imports.into_iter().collect();
        sorted_imports.sort();
        for imp in sorted_imports {
            final_pytest_content.push_str(&imp);
            final_pytest_content.push('\n');
        }
        final_pytest_content.push('\n');

        for func_str in all_pytest_functions {
            final_pytest_content.push_str(&func_str);
            final_pytest_content.push_str("\n\n");
        }
        final_pytest_content.push_str("if __name__ == \"__main__\":\n");
        final_pytest_content.push_str("    pytest.main()\n");

        // Changed the output file name to have a more explicit "test_" prefix
        let pytest_output_path = Path::new("test_generated_suite.py");
        if let Err(e) =
            file_handler::write_content_to_file(pytest_output_path, &final_pytest_content)
        {
            verbose_eprintln!(
                quiet_mode,
                "[ERROR] Failed to write consolidated pytest file ({}): {}",
                pytest_output_path.display(),
                e
            );
            return Err(AppError::Io(e));
        } else {
            verbose_println!(
                quiet_mode,
                "\n[INFO] All tests written to {}",
                pytest_output_path.display()
            );
        }
    } else if !ast_func_defs.is_empty() {
        verbose_println!(
            quiet_mode,
            "\n[INFO] No tests were generated for any function."
        );
    }

    if quiet_mode {
        println!("Done.");
    } else {
        // This message goes to stdout, not the log file.
        println!(
            "\nTest generation process finished. See 'testgen.log' for verbose output and 'cfg_details.log' for CFG information."
        );
    }

    Ok(())
}
