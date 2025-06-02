mod ast_loader;
mod cfg;
mod path;
mod testgen;

use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python_file_path_str = "./test_file.py"; // Assuming this is your target Python file
    let python_file_path = Path::new(python_file_path_str);

    let python_module_name = python_file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown_module");

    println!("Loading AST from: {}", python_file_path_str);
    let ast_func_defs = ast_loader::load_ast_from_file(python_file_path_str)?;

    if ast_func_defs.is_empty() {
        println!("No function definitions found in {}.", python_file_path_str);
        return Ok(());
    }

    for func_def in ast_func_defs {
        let func_name = func_def.name.to_string();
        println!("\nProcessing function: {}", func_name);

        let mut cfg_data = cfg::ControlFlowGraph::new();
        cfg_data.from_ast(func_def); // func_def is moved here

        // Optional: Print discovered paths from CFG
        // path::PathScraper::print_paths(&cfg_data);

        if let Some(paths) = path::PathScraper::get_paths(&cfg_data) {
            if paths.is_empty() {
                println!("No executable paths found for function {}.", func_name);
                continue;
            }

            println!(
                "Analyzing {} paths for function {}...",
                paths.len(),
                func_name
            );
            let path_constraint_results = path::analyze_paths(&paths, &cfg_data);

            path::print_paths(&cfg_data, &paths); // This calls analyze_paths again.
            path::PathScraper::print_paths(&cfg_data); // Print paths again after analysis

            // Generate Pytest tests
            println!("Generating Pytest code for {}...", func_name);
            let pytest_string = testgen::PytestGenerator::generate_pytest_file_string(
                &func_name,
                &path_constraint_results,
                &paths,
                &cfg_data,
                Some(python_module_name), // Pass the module name for imports
            );

            let test_file_name = format!("test_{}.py", func_name.to_lowercase().replace(" ", "_"));
            match fs::write(&test_file_name, pytest_string) {
                Ok(_) => println!("Successfully wrote tests to {}", test_file_name),
                Err(e) => eprintln!("Failed to write test file {}: {}", test_file_name, e),
            }
        } else {
            println!("No paths found by PathScraper for function {}.", func_name);
        }
    }
    Ok(())
}
