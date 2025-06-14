use crate::cfg::{ControlFlowGraph, Edge, Node, NodeId};
use crate::path::PathConstraintResult;
use rustpython_ast::{Constant, Expr, ExprConstant, ExprName};
use std::collections::HashMap;

pub struct PytestGenerator;

impl PytestGenerator {
    /// Parses the Z3 model string into a map of variable assignments.
    /// Handles common `(define-fun var () Type val)` and `var -> val` formats.
    fn parse_z3_model(model_str: &str) -> HashMap<String, String> {
        let mut assignments = HashMap::new();
        for line in model_str.lines() {
            let trimmed_line = line.trim();
            if trimmed_line.starts_with("(define-fun") && trimmed_line.ends_with(")") {
                let parts: Vec<&str> = trimmed_line
                    .strip_prefix("(define-fun ")
                    .unwrap_or("")
                    .strip_suffix(")")
                    .unwrap_or("")
                    .split_whitespace()
                    .collect();

                if parts.len() >= 4 && parts[1] == "()" {
                    let name = parts[0].to_string();
                    let value_str = if parts[3] == "(-"
                        && parts.len() >= 5
                        && parts[4].ends_with(")")
                    {
                        format!("-{}", parts[4].strip_suffix(")").unwrap_or(parts[4]))
                    } else if parts[3].starts_with("(") && parts[3].contains("-") && parts.len() > 4
                    {
                        // e.g. (bvneg (_ bv2 32))
                        // This is a placeholder for more complex Z3 values, may need specific parsing
                        parts[3..].join(" ")
                    } else {
                        parts[3].to_string()
                    };
                    assignments.insert(name, value_str);
                }
            } else if trimmed_line.contains("->") {
                let parts: Vec<&str> = trimmed_line.split("->").map(str::trim).collect();
                if parts.len() == 2 {
                    assignments.insert(parts[0].to_string(), parts[1].to_string());
                }
            }
        }
        assignments
    }

    /// Converts a rustpython_ast::Constant to its Python string representation.
    fn format_python_constant(constant: &Constant) -> String {
        match constant {
            Constant::Int(i) => i.to_string(),
            Constant::Bool(b) => if *b { "True" } else { "False" }.to_string(),
            Constant::Str(s) => {
                // Basic string formatting, escaping double quotes and backslashes
                format!(
                    "\"{}\"",
                    s.to_string().replace('\\', "\\\\").replace('\"', "\\\"")
                )
            }
            Constant::Float(f) => f.to_string(),
            Constant::Complex { real, imag } => format!("complex({}, {})", real, imag),
            Constant::None => "None".to_string(),
            Constant::Ellipsis => "Ellipsis".to_string(),
            Constant::Bytes(b_val) => {
                let mut repr = String::from("b\"");
                for &byte_char in b_val {
                    if byte_char == b'\"' {
                        repr.push_str("\\\"");
                    } else if byte_char == b'\\' {
                        repr.push_str("\\\\");
                    } else if byte_char >= 32 && byte_char < 127 {
                        repr.push(byte_char as char);
                    } else {
                        repr.push_str(&format!("\\x{:02x}", byte_char));
                    }
                }
                repr.push('\"');
                repr
            }
            _ => {
                format!("# Unsupported constant value: {:?}", constant)
            }
        }
    }

    /// Generates a single pytest test function as a string.
    fn generate_test_function_string(
        original_function_name: &str,
        path_index: usize,
        model_str: &str,
        path: &[(NodeId, Edge)],
        cfg: &ControlFlowGraph,
    ) -> Option<String> {
        let model_assignments = Self::parse_z3_model(model_str);
        let mut args_list = Vec::new();

        // Iterate through the original function arguments to ensure all are present
        for (arg_name, type_hint_opt) in cfg.get_arguments() {
            let py_value = if let Some(model_value_str) = model_assignments.get(arg_name) {
                // Value found in Z3 model
                match model_value_str.as_str() {
                    "true" => "True".to_string(),
                    "false" => "False".to_string(),
                    _ => model_value_str.clone(),
                }
            } else {
                // Value not in Z3 model, use placeholder based on type hint
                if let Some(type_hint) = type_hint_opt {
                    match type_hint.as_str() {
                        "int" => "0".to_string(),
                        "bool" => "False".to_string(),
                        "str" => "\"\"".to_string(),
                        _ => "None".to_string(), // Default placeholder for other types
                    }
                } else {
                    "None".to_string() // Default placeholder if no type hint
                }
            };
            args_list.push(format!("{}={}", arg_name, py_value));
        }

        let func_args_str = args_list.join(", ");

        let (terminal_node_id, _edge_from_terminal) = match path.last() {
            Some(last_step) => last_step,
            None => return None,
        };

        let terminal_node = cfg.get_node(*terminal_node_id)?;
        let mut test_body_lines = Vec::new();
        let call_stmt = format!("{}({})", original_function_name, func_args_str);

        match terminal_node {
            Node::Return {
                stmt: opt_expr_stmt,
            } => {
                if let Some(expr) = &opt_expr_stmt.value {
                    match expr.as_ref() {
                        Expr::Constant(ExprConstant {
                            value: const_val, ..
                        }) => {
                            let expected_value_str = Self::format_python_constant(const_val);
                            test_body_lines.push(format!(
                                "    assert {} == {}",
                                call_stmt, expected_value_str
                            ));
                        }
                        _ => {
                            test_body_lines.push(format!(
                                "    # Path returns a non-constant expression: {:?}",
                                expr
                            ));
                            test_body_lines.push(format!("    returnValue = {}", call_stmt));
                            test_body_lines.push(
                                "    # TODO: Add manual assertion for returnValue".to_string(),
                            );
                        }
                    }
                } else {
                    test_body_lines.push(format!("    assert {} is None", call_stmt));
                }
            }
            Node::Raise { stmt: opt_exc_stmt } => {
                if let Some(exc_expr) = &opt_exc_stmt.exc {
                    match exc_expr.as_ref() {
                        Expr::Name(ExprName { id, .. }) => {
                            test_body_lines.push(format!("    with pytest.raises({}):", id));
                            test_body_lines.push(format!("        {}", call_stmt));
                        }
                        _ => {
                            test_body_lines.push(format!(
                                "    # Path raises a non-Name exception: {:?}",
                                exc_expr
                            ));
                            test_body_lines.push(
                                "    with pytest.raises(Exception): # Generic check".to_string(),
                            );
                            test_body_lines.push(format!("        {}", call_stmt));
                        }
                    }
                } else {
                    test_body_lines.push("    # Path involves a bare 'raise'".to_string());
                    test_body_lines.push("    with pytest.raises(Exception): # Generic check for any re-raised exception".to_string());
                    test_body_lines.push(format!("        {}", call_stmt));
                }
            }
            Node::Cond { .. } => return None,
        }

        if test_body_lines.is_empty() {
            return None;
        }

        let test_function_name = format!(
            "test_{}_path_{}",
            original_function_name.replace(" ", "_").to_lowercase(),
            path_index
        );
        let mut fn_string = format!("def {}():\n", test_function_name);
        for line in test_body_lines {
            fn_string.push_str(&line);
            fn_string.push('\n');
        }
        Some(fn_string)
    }

    /// Generates the full content of a pytest file as a string.
    pub fn generate_pytest_file_string(
        original_function_name: &str,
        path_results: &[PathConstraintResult],
        all_paths: &Vec<Vec<(NodeId, Edge)>>,
        cfg: &ControlFlowGraph,
        module_name_for_import: Option<&str>,
    ) -> String {
        let mut test_file_content = String::new();
        test_file_content.push_str("import pytest\n");

        if let Some(module_name) = module_name_for_import {
            test_file_content.push_str(&format!(
                "from {} import {}\n",
                module_name, original_function_name
            ));
        } else {
            test_file_content.push_str(&format!(
                "# Ensure '{}' is importable or defined in the test environment\n",
                original_function_name
            ));
        }
        test_file_content.push_str("\n");

        let mut generated_tests_count = 0;
        for path_result in path_results {
            if path_result.is_satisfiable {
                if let Some(model_str) = &path_result.model {
                    if let Some(current_path_nodes_edges) = all_paths.get(path_result.path_index) {
                        if !current_path_nodes_edges.is_empty() {
                            if let Some(test_fn_str) = Self::generate_test_function_string(
                                original_function_name,
                                path_result.path_index,
                                model_str,
                                current_path_nodes_edges,
                                cfg, // Pass the cfg here
                            ) {
                                test_file_content.push_str(&test_fn_str);
                                test_file_content.push_str("\n\n");
                                generated_tests_count += 1;
                            }
                        }
                    }
                }
            }
        }

        if generated_tests_count == 0 {
            test_file_content.push_str(&format!(
                "# No satisfiable paths found or no tests could be generated for function '{}'.\n",
                original_function_name
            ));
        }

        test_file_content.push_str("\nif __name__ == \"__main__\":\n");
        test_file_content.push_str("    pytest.main()\n");

        test_file_content
    }
}
