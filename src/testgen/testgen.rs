use crate::cfg::{ControlFlowGraph, Edge, Node, NodeId};
use crate::path::PathConstraintResult;
use rustpython_ast::{Constant, Expr, ExprCall, ExprConstant, ExprName};
use std::collections::HashMap;
use std::fmt::Write;

pub struct PytestGenerator;

const DEFAULT_PY_NONE: &str = "None";
const DEFAULT_PY_INT: &str = "0";
const DEFAULT_PY_BOOL: &str = "False";
const DEFAULT_PY_STR: &str = "\"\"";

/// Holds the generated imports and test functions for a single Python function.
pub struct GeneratedTestSuite {
    pub imports: Vec<String>,
    pub test_functions: Vec<String>,
}

impl PytestGenerator {
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

    fn generate_test_function_string(
        original_function_name: &str,
        path_index: usize,
        model_str: &str,
        path: &[(NodeId, Edge)],
        cfg: &ControlFlowGraph,
    ) -> Option<String> {
        let model_assignments = Self::parse_z3_model(model_str);
        
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
                        type_hint_opt
                            .as_ref()
                            .map_or(DEFAULT_PY_NONE.to_string(), |type_hint| {
                                match type_hint.as_str() {
                                    "int" => DEFAULT_PY_INT.to_string(),
                                    "bool" => DEFAULT_PY_BOOL.to_string(),
                                    "str" => DEFAULT_PY_STR.to_string(),
                                    _ => DEFAULT_PY_NONE.to_string(),
                                }
                            })
                    });
                format!("{}={}", arg_name, py_value)
            })
            .collect::<Vec<_>>()
            .join(", ");

        let (terminal_node_id, _edge_from_terminal) = path.last()?;
        let terminal_node = cfg.get_node(*terminal_node_id)?;

        let mut test_body_lines = Vec::new();
        let call_stmt = format!("{}({})", original_function_name, func_args_str);

        match terminal_node {
            Node::Return { stmts: _, stmt: return_stmt } => {
                if let Some(expr_box) = &return_stmt.value {
                    match expr_box.as_ref() {
                        Expr::Constant(ExprConstant { value: const_val, .. }) => {
                            let expected_value_str = Self::format_python_constant(const_val);
                            test_body_lines.push(format!("    assert {} == {}", call_stmt, expected_value_str));
                        }
                        Expr::Name(ExprName { id, .. }) => {
                            let returned_var_name = id.to_string();
                            let mut best_ssa_key: Option<String> = None;
                            let mut max_ssa_index: i32 = -1;

                            // Find the latest SSA version of the returned variable in the model
                            for (key, _value) in model_assignments.iter() {
                                if key.starts_with(&returned_var_name) {
                                    // Check for SSA pattern like "var_assigned_type!index"
                                    // Example: key = "mode_assigned_str!1", returned_var_name = "mode"
                                    // Prefix to check: "mode_assigned_"
                                    let ssa_prefix_pattern = format!("{}_assigned_", returned_var_name);
                                    if key.starts_with(&ssa_prefix_pattern) {
                                        if let Some(pos_bang) = key.rfind('!') {
                                            if pos_bang > ssa_prefix_pattern.len() -1 { // Ensure '!' is after the type part
                                                let index_str = &key[(pos_bang + 1)..];
                                                if let Ok(current_index) = index_str.parse::<i32>() {
                                                    if current_index > max_ssa_index {
                                                        max_ssa_index = current_index;
                                                        best_ssa_key = Some(key.clone());
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            let mut found_value_for_assertion = false;
                            if let Some(key_to_use) = best_ssa_key {
                                if let Some(model_value_str) = model_assignments.get(&key_to_use) {
                                    let python_model_value = match model_value_str.as_str() {
                                        "true" => "True".to_string(),
                                        "false" => "False".to_string(),
                                        _ => model_value_str.clone(), // Assumes it's already a Python literal string
                                    };
                                    test_body_lines.push(format!("    returnValue = {}", call_stmt));
                                    test_body_lines.push(format!("    assert returnValue == {}", python_model_value));
                                    found_value_for_assertion = true;
                                }
                            }
                            
                            // Fallback: if no SSA version was found, try the original variable name
                            // (e.g., an input argument returned directly without reassignment).
                            if !found_value_for_assertion {
                                if let Some(model_value_str) = model_assignments.get(&returned_var_name) {
                                     let python_model_value = match model_value_str.as_str() {
                                        "true" => "True".to_string(),
                                        "false" => "False".to_string(),
                                        _ => model_value_str.clone(),
                                    };
                                    test_body_lines.push(format!("    returnValue = {}", call_stmt));
                                    test_body_lines.push(format!("    assert returnValue == {}", python_model_value));
                                    found_value_for_assertion = true;
                                }
                            }

                            if !found_value_for_assertion {
                                test_body_lines.push(format!("    # Path returns variable '{}' whose value (or its SSA version) is not in the Z3 model for this path.", returned_var_name));
                                test_body_lines.push(format!("    returnValue = {}", call_stmt));
                                test_body_lines.push("    # TODO: Add manual assertion for returnValue".to_string());
                            }
                        }
                        _ => {
                            test_body_lines.push(format!("    # Path returns a non-constant expression: {:?}", expr_box));
                            test_body_lines.push(format!("    returnValue = {}", call_stmt));
                            test_body_lines.push("    # TODO: Add manual assertion for returnValue".to_string());
                        }
                    }
                } else {
                    test_body_lines.push(format!("    assert {} is None", call_stmt));
                }
            }
            Node::Raise { stmts: _, stmt: raise_stmt } => {
                let exception_name = if let Some(exc_expr_box) = &raise_stmt.exc {
                    match exc_expr_box.as_ref() {
                        Expr::Name(ExprName { id, .. }) => id.to_string(),
                        Expr::Call(ExprCall { func, .. }) => {
                            if let Expr::Name(ExprName { id, .. }) = func.as_ref() {
                                id.to_string()
                            } else {
                                "Exception".to_string()
                            }
                        }
                        _ => "Exception".to_string(),
                    }
                } else {
                    "Exception".to_string()
                };

                if exception_name == "Exception" && raise_stmt.exc.is_some() {
                    test_body_lines.push(format!("    # Path raises a non-standard or complex exception: {:?}", raise_stmt.exc));
                } else if raise_stmt.exc.is_none() {
                     test_body_lines.push("    # Path involves a bare 'raise'".to_string());
                }

                test_body_lines.push(format!("    with pytest.raises({}):", exception_name));
                test_body_lines.push(format!("        {}", call_stmt));
            }
            Node::Cond { .. } => return None,
        }

        if test_body_lines.is_empty() {
            return None; 
        }
        
        let test_function_name = format!(
            "test_{}_path_{}",
            original_function_name.to_lowercase().replace(|c: char| !c.is_alphanumeric() && c != '_', "_"), 
            path_index
        );
        
        let mut fn_string = String::with_capacity(100 + test_body_lines.join("\n").len());
        writeln!(fn_string, "def {}():", test_function_name).ok()?; 
        for line in test_body_lines {
            writeln!(fn_string, "{}", line).ok()?;
        }
        Some(fn_string)
    }

    /// Generates imports and test function strings for a single Python function.
    pub fn generate_suite_for_function(
        original_function_name: &str,
        path_results: &[PathConstraintResult],
        all_paths: &[Vec<(NodeId, Edge)>],
        cfg: &ControlFlowGraph,
        module_name_for_import: Option<&str>,
    ) -> GeneratedTestSuite {
        let mut imports = Vec::new();
        let mut test_functions = Vec::new();

        if let Some(module_name) = module_name_for_import {
            imports.push(format!(
                "from {} import {}",
                module_name, original_function_name
            ));
        } else {
            imports.push(format!(
                "# Ensure '{}' is importable or defined in the test environment",
                original_function_name
            ));
        }

        let mut generated_any_test_for_this_func = false;
        for path_result in path_results.iter().filter(|pr| pr.is_satisfiable) {
            if let Some(model_str) = &path_result.model {
                if let Some(current_path_nodes_edges) = all_paths.get(path_result.path_index) {
                    if !current_path_nodes_edges.is_empty() {
                        if let Some(test_fn_str) = Self::generate_test_function_string(
                            original_function_name,
                            path_result.path_index,
                            model_str,
                            current_path_nodes_edges,
                            cfg,
                        ) {
                            test_functions.push(test_fn_str);
                            generated_any_test_for_this_func = true;
                        }
                    }
                }
            }
        }

        if !generated_any_test_for_this_func && !path_results.is_empty() {
            // Add a comment if no tests were generated for this specific Python function
            // but paths were analyzed.
            test_functions.push(format!(
                "\n# No satisfiable paths led to test generation for function '{}'.\n",
                original_function_name
            ));
        }

        GeneratedTestSuite {
            imports,
            test_functions,
        }
    }
}
