use crate::cfg::{ControlFlowGraph, Edge, Node, NodeId};
use crate::path::PathConstraintResult;
// Added ExprCall for more detailed exception parsing
use rustpython_ast::{Constant, Expr, ExprCall, ExprConstant, ExprName};
use std::collections::HashMap;
use std::fmt::Write; // Used for building strings efficiently

/// `PytestGenerator` is responsible for generating pytest test file content
/// based on analyzed execution paths of a Python function.
pub struct PytestGenerator;

impl PytestGenerator {
    /// Parses a Z3 model string into a map of variable assignments.
    /// Made `pub(crate)` to be accessible from main.rs for formatted model printing.
    ///
    /// The Z3 model string typically contains lines defining functions (variables)
    /// or direct mappings. This function handles common formats like:
    /// - `(define-fun var_name () VarType value)`
    /// - `var_name -> value`
    ///
    /// # Arguments
    /// * `model_str`: A string slice representing the Z3 model output.
    ///
    /// # Returns
    /// A `HashMap` where keys are variable names and values are their string representations.
    pub(crate) fn parse_z3_model(model_str: &str) -> HashMap<String, String> {
        // Replaced manual loop with iterator chain for conciseness and idiomatic Rust.
        model_str
            .lines()
            .filter_map(|line| {
                let trimmed_line = line.trim();
                if trimmed_line.starts_with("(define-fun") && trimmed_line.ends_with(')') {
                    // Using unwrap_or_default which results in "" for &str if prefix/suffix is not found.
                    // This avoids panic and keeps similar logic to original unwrap_or("").
                    let core_parts: Vec<&str> = trimmed_line
                        .strip_prefix("(define-fun ")
                        .unwrap_or_default()
                        .strip_suffix(')')
                        .unwrap_or_default()
                        .split_whitespace()
                        .collect();

                    if core_parts.len() >= 4 && core_parts[1] == "()" {
                        let name = core_parts[0].to_string();
                        // Simplified negative number parsing and complex value handling.
                        let value_str = if core_parts[3] == "(-"
                            && core_parts.len() >= 5
                            && core_parts[4].ends_with(')')
                        {
                            // Handles "(- number)" format
                            format!("-{}", core_parts[4].strip_suffix(')').unwrap_or(core_parts[4]))
                        } else if core_parts[3].starts_with('(') && core_parts[3].contains('-') && core_parts.len() > 4 {
                            // Placeholder for more complex Z3 values, e.g. (bvneg (_ bv2 32))
                            core_parts[3..].join(" ")
                        } else {
                            core_parts[3].to_string()
                        };
                        Some((name, value_str))
                    } else {
                        None
                    }
                } else if trimmed_line.contains("->") {
                    let parts: Vec<&str> = trimmed_line.split("->").map(str::trim).collect();
                    if parts.len() == 2 {
                        Some((parts[0].to_string(), parts[1].to_string()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Converts a `rustpython_ast::Constant` to its Python string representation.
    ///
    /// # Arguments
    /// * `constant`: A reference to the `Constant` to format.
    ///
    /// # Returns
    /// A `String` representing the Python literal.
    fn format_python_constant(constant: &Constant) -> String {
        match constant {
            Constant::Int(i) => i.to_string(),
            Constant::Bool(b) => if *b { "True" } else { "False" }.to_string(),
            Constant::Str(s_val) => {
                // Escapes backslashes and double quotes for Python string literals.
                format!("\"{}\"", s_val.replace('\\', "\\\\").replace('\"', "\\\""))
            }
            Constant::Float(f) => f.to_string(),
            Constant::Complex { real, imag } => format!("complex({}, {})", real, imag),
            Constant::None => "None".to_string(),
            Constant::Ellipsis => "...".to_string(), // Changed to "..." for standard Ellipsis literal
            Constant::Bytes(b_val) => {
                // Efficiently builds the bytes literal string.
                let mut repr = String::with_capacity(b_val.len() * 2 + 3); // Pre-allocate for b"" and escapes
                repr.push_str("b\"");
                for &byte_char in b_val {
                    match byte_char {
                        b'"' => repr.push_str("\\\""),
                        b'\\' => repr.push_str("\\\\"),
                        // Check for printable ASCII characters
                        32..=126 => repr.push(byte_char as char),
                        // Escape non-printable characters
                        _ => write!(repr, "\\x{:02x}", byte_char).unwrap(), // Should not fail with String
                    }
                }
                repr.push('\"');
                repr
            }
            _ => {
                // Catch-all for unsupported Constant types.
                // This should not happen with the current rustpython-ast version.
                format!("Unsupported constant type: {:?}", constant)
            }
            // As per rustpython-ast 0.4.0, Constant::Tuple was removed.
            // If other Constant variants are added in the future, this match will need to be updated.
            // The previous refactoring added a catch-all `_` which is fine.
            // For now, assuming the provided variants are exhaustive for the current library version.
        }
    }

    /// Generates a single pytest test function as a string.
    ///
    /// # Arguments
    /// * `original_function_name`: Name of the Python function under test.
    /// * `path_index`: Index of the current path, used for naming the test function.
    /// * `model_str`: Z3 model string for the current path.
    /// * `path`: The sequence of (NodeId, Edge) representing the current execution path.
    /// * `cfg`: The `ControlFlowGraph` of the function.
    ///
    /// # Returns
    /// An `Option<String>` containing the test function string, or `None` if a test cannot be generated.
    fn generate_test_function_string(
        original_function_name: &str,
        path_index: usize,
        model_str: &str,
        path: &[(NodeId, Edge)],
        cfg: &ControlFlowGraph,
    ) -> Option<String> {
        let model_assignments = Self::parse_z3_model(model_str);
        
        // Build the argument list string for the function call.
        // Uses placeholders for arguments not found in the Z3 model.
        let func_args_str = cfg
            .get_arguments()
            .iter()
            .map(|(arg_name, type_hint_opt)| {
                let py_value = model_assignments
                    .get(arg_name)
                    .map(|model_value_str| match model_value_str.as_str() {
                        "true" => "True".to_string(),
                        "false" => "False".to_string(),
                        _ => model_value_str.clone(),
                    })
                    .unwrap_or_else(|| {
                        // Determine placeholder based on type hint.
                        type_hint_opt
                            .as_ref()
                            .map_or("None".to_string(), |type_hint| {
                                match type_hint.as_str() {
                                    "int" => "0".to_string(),
                                    "bool" => "False".to_string(),
                                    "str" => "\"\"".to_string(),
                                    _ => "None".to_string(), // Default placeholder
                                }
                            })
                    });
                format!("{}={}", arg_name, py_value)
            })
            .collect::<Vec<_>>()
            .join(", ");

        // Get the terminal node of the path.
        let (terminal_node_id, _edge_from_terminal) = path.last()?; // Use '?' for early return if path is empty.
        let terminal_node = cfg.get_node(*terminal_node_id)?; // Use '?' for early return if node not found.

        let mut test_body_lines = Vec::new();
        let call_stmt = format!("{}({})", original_function_name, func_args_str);

        // Generate assertions based on the terminal node type (Return or Raise).
        match terminal_node {
            Node::Return { stmts: _, stmt: return_stmt } => {
                if let Some(expr_box) = &return_stmt.value {
                    match expr_box.as_ref() {
                        Expr::Constant(ExprConstant { value: const_val, .. }) => {
                            let expected_value_str = Self::format_python_constant(const_val);
                            test_body_lines.push(format!("    assert {} == {}", call_stmt, expected_value_str));
                        }
                        _ => {
                            // Handle non-constant return expressions.
                            test_body_lines.push(format!("    # Path returns a non-constant expression: {:?}", expr_box));
                            test_body_lines.push(format!("    returnValue = {}", call_stmt));
                            test_body_lines.push("    # TODO: Add manual assertion for returnValue".to_string());
                        }
                    }
                } else {
                    // Handle `return None`.
                    test_body_lines.push(format!("    assert {} is None", call_stmt));
                }
            }
            Node::Raise { stmts: _, stmt: raise_stmt } => {
                // Refactored to assert specific native Python exceptions if identifiable.
                let exception_name = if let Some(exc_expr_box) = &raise_stmt.exc {
                    match exc_expr_box.as_ref() {
                        Expr::Name(ExprName { id, .. }) => id.to_string(), // e.g., raise ValueError
                        Expr::Call(ExprCall { func, .. }) => { // e.g., raise ValueError("message")
                            if let Expr::Name(ExprName { id, .. }) = func.as_ref() {
                                id.to_string()
                            } else {
                                "Exception".to_string() // Fallback for complex calls like raise some_func()
                            }
                        }
                        _ => "Exception".to_string(), // Fallback for other expression types
                    }
                } else {
                    "Exception".to_string() // Fallback for bare raise
                };

                if exception_name == "Exception" && raise_stmt.exc.is_some() {
                     // If it's a generic Exception but there was an expression, note it.
                    test_body_lines.push(format!("    # Path raises a non-standard or complex exception: {:?}", raise_stmt.exc));
                } else if raise_stmt.exc.is_none() {
                     test_body_lines.push("    # Path involves a bare 'raise'".to_string());
                }


                test_body_lines.push(format!("    with pytest.raises({}):", exception_name));
                test_body_lines.push(format!("        {}", call_stmt));
            }
            Node::Cond { .. } => return None, // Paths should not end on a Cond node.
        }

        if test_body_lines.is_empty() {
            return None; // No assertion lines generated.
        }

        // Format the test function name.
        let test_function_name = format!(
            "test_{}_path_{}",
            original_function_name.replace(' ', "_").to_lowercase(), // Pythonic names
            path_index
        );
        
        // Build the test function string.
        // Using String::with_capacity for slight performance improvement by pre-allocating.
        let mut fn_string = String::with_capacity(100 + test_body_lines.join("\n").len());
        writeln!(fn_string, "def {}():", test_function_name).ok()?; // Use writeln for conciseness
        for line in test_body_lines {
            writeln!(fn_string, "{}", line).ok()?;
        }
        Some(fn_string)
    }

    /// Generates the full content of a pytest file as a string.
    ///
    /// # Arguments
    /// * `original_function_name`: Name of the Python function for which tests are generated.
    /// * `path_results`: A slice of `PathConstraintResult` from Z3 analysis.
    /// * `all_paths`: A vector of all identified execution paths.
    /// * `cfg`: The `ControlFlowGraph` of the function.
    /// * `module_name_for_import`: Optional name of the module for the import statement.
    ///
    /// # Returns
    /// A `String` containing the complete pytest file content.
    pub fn generate_pytest_file_string(
        original_function_name: &str,
        path_results: &[PathConstraintResult],
        all_paths: &[Vec<(NodeId, Edge)>], // Changed to slice for flexibility
        cfg: &ControlFlowGraph,
        module_name_for_import: Option<&str>,
    ) -> String {
        // Using String::with_capacity for slight performance improvement.
        let mut test_file_content = String::with_capacity(1024); // Initial capacity guess
        
        // Add standard imports.
        test_file_content.push_str("import pytest\n");

        // Add import for the function under test.
        if let Some(module_name) = module_name_for_import {
            writeln!(test_file_content, "from {} import {}", module_name, original_function_name).unwrap();
        } else {
            writeln!(test_file_content, "# Ensure '{}' is importable or defined in the test environment", original_function_name).unwrap();
        }
        test_file_content.push('\n');

        let mut generated_tests_count = 0;
        // Iterate over satisfiable paths and generate test functions.
        // Replaced manual loop with filter_map and for_each for a more functional style.
        path_results
            .iter()
            .filter(|pr| pr.is_satisfiable) // Process only satisfiable paths
            .filter_map(|path_result| {
                // Chain Optionals: model, path, then test_function_string
                path_result.model.as_ref().and_then(|model_str| {
                    all_paths.get(path_result.path_index).and_then(|current_path_nodes_edges| {
                        if current_path_nodes_edges.is_empty() {
                            None
                        } else {
                            Self::generate_test_function_string(
                                original_function_name,
                                path_result.path_index,
                                model_str,
                                current_path_nodes_edges,
                                cfg,
                            )
                        }
                    })
                })
            })
            .for_each(|test_fn_str| {
                test_file_content.push_str(&test_fn_str);
                test_file_content.push_str("\n\n"); // Two newlines between functions
                generated_tests_count += 1;
            });


        if generated_tests_count == 0 {
            writeln!(test_file_content, "# No satisfiable paths found or no tests could be generated for function '{}'.", original_function_name).unwrap();
        }

        // Add the main block to run pytest.
        test_file_content.push_str("\nif __name__ == \"__main__\":\n");
        test_file_content.push_str("    pytest.main()\n");

        test_file_content
    }
}
