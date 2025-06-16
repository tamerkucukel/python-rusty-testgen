use crate::cfg::{ControlFlowGraph, Edge, Node, NodeId};
use crate::path::PathConstraintResult;
use rustpython_ast::{Constant, Expr, ExprCall, ExprConstant, ExprName};
use std::collections::HashMap;
use std::fmt::Write;
use std::ops::Deref;

// Helper function to convert Z3 real string representations to Python float literals
fn z3_real_to_python_float_literal(z3_val: &str) -> String {
    let mut s = z3_val.trim().to_string();
    if s.ends_with('?') {
        // Remove Z3's optional '?' suffix for reals
        s.pop();
    }

    let is_negative = s.starts_with("(- ") && s.ends_with(')');
    let mut effective_s = if is_negative {
        // Strip "(- " prefix and ")" suffix, then trim
        s.strip_prefix("(- ")
            .unwrap_or(&s) // Should not fail if is_negative is true
            .strip_suffix(')')
            .unwrap_or(&s) // Should not fail if is_negative is true
            .trim()
            .to_string()
    } else {
        s // Use the original string (already trimmed and '?' removed)
    };

    // Check for fractional form like "(/ numerator denominator)"
    if effective_s.starts_with("(/ ") && effective_s.ends_with(')') {
        let fraction_content = effective_s
            .strip_prefix("(/ ")
            .unwrap_or("")
            .strip_suffix(')')
            .unwrap_or("")
            .trim();

        // Split the numerator and denominator.
        // This assumes they are separated by whitespace and are simple numbers
        // or recursively parsable by this function.
        let mut parts_iter = fraction_content.split_whitespace();
        if let (Some(num_str_raw), Some(den_str_raw)) = (parts_iter.next(), parts_iter.next()) {
            // Further splitting or parsing might be needed if num/den are complex S-expressions.
            // For now, assume they are directly parsable or simple Z3 reals.
            let num_cleaned = z3_real_to_python_float_literal(num_str_raw);
            let den_cleaned = z3_real_to_python_float_literal(den_str_raw);

            if let (Ok(num_f64), Ok(den_f64)) =
                (num_cleaned.parse::<f64>(), den_cleaned.parse::<f64>())
            {
                if den_f64 != 0.0 {
                    let result = num_f64 / den_f64;
                    // Format to a string with sufficient precision, then clean up.
                    let mut formatted_result = format!("{:.17}", result); // f64 has ~15-17 decimal digits precision
                    if formatted_result.contains('.') {
                        formatted_result = formatted_result.trim_end_matches('0').to_string();
                        if formatted_result.ends_with('.') {
                            formatted_result.push('0'); // Ensure "1." becomes "1.0"
                        }
                    }
                    effective_s = formatted_result;
                } else {
                    // Division by zero, return original fraction form (or an error indicator)
                    // For now, we return the (potentially negative) original S-expression string
                    return if is_negative {
                        format!("(- {})", effective_s)
                    } else {
                        effective_s
                    };
                }
            } else {
                // Failed to parse numerator or denominator as f64
                return if is_negative {
                    format!("(- {})", effective_s)
                } else {
                    effective_s
                };
            }
        } else {
            // Malformed fraction string
            return if is_negative {
                format!("(- {})", effective_s)
            } else {
                effective_s
            };
        }
    }
    // If not a fraction, effective_s is now a cleaned decimal or integer string

    if is_negative {
        format!("-{}", effective_s)
    } else {
        effective_s
    }
}

pub struct PytestGenerator;

const DEFAULT_PY_NONE: &str = "None";
const DEFAULT_PY_INT: &str = "0";
const DEFAULT_PY_BOOL: &str = "False";
const DEFAULT_PY_STR: &str = "\"\"";
const DEFAULT_PY_FLOAT: &str = "0.0";

/// Holds the generated imports and test functions for a single Python function.
pub struct GeneratedTestSuite {
    pub imports: Vec<String>,
    pub test_functions: Vec<String>,
}

impl PytestGenerator {
    pub(crate) fn parse_z3_model(model_str: &str) -> HashMap<String, String> {
        model_str
            .lines()
            .filter_map(|line| {
                let trimmed_line = line.trim();
                if trimmed_line.starts_with("(define-fun") && trimmed_line.ends_with(')') {
                    let core_parts: Vec<&str> = trimmed_line
                        .strip_prefix("(define-fun ")
                        .unwrap_or_default()
                        .strip_suffix(')')
                        .unwrap_or_default()
                        .split_whitespace()
                        .collect();

                    if core_parts.len() >= 4 && core_parts[1] == "()" {
                        let name = core_parts[0].to_string();
                        // Reconstruct the full value expression string from Z3 model
                        // It starts after name, "()", and type, so from core_parts[3]
                        let raw_value_str = core_parts
                            .iter()
                            .skip(3)
                            .copied()
                            .collect::<Vec<&str>>()
                            .join(" ");

                        let python_literal_value = z3_real_to_python_float_literal(&raw_value_str);
                        Some((name, python_literal_value))
                    } else {
                        None
                    }
                } else if trimmed_line.contains("->") {
                    // Handle simpler "var -> val" if present
                    let parts: Vec<&str> = trimmed_line.split("->").map(str::trim).collect();
                    if parts.len() == 2 {
                        let name = parts[0].to_string();
                        let raw_value_str = parts[1].to_string();
                        let python_literal_value = z3_real_to_python_float_literal(&raw_value_str);
                        Some((name, python_literal_value))
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
                format!(
                    "\"{}\"",
                    s_val.as_str().replace('\\', "\\\\").replace('\"', "\\\"")
                ) // Use as_str()
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
                // For any other constant type, we return a placeholder.
                // This should not happen in well-formed ASTs.
                format!("UnsupportedConstant({:?})", constant)
            }
        }
    }

    fn generate_test_function_string(
        original_function_name: &str,
        path_index: usize,
        model_str: &str,
        path: &[(NodeId, Edge)],
        cfg: &ControlFlowGraph, // ControlFlowGraph contains return type hint
    ) -> Option<String> {
        let model_assignments = Self::parse_z3_model(model_str);

        let func_args_str =
            cfg.get_arguments()
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
                            type_hint_opt.as_ref().map_or(
                                DEFAULT_PY_NONE.to_string(),
                                |type_hint| match type_hint.as_str() {
                                    "int" => DEFAULT_PY_INT.to_string(),
                                    "bool" => DEFAULT_PY_BOOL.to_string(),
                                    "str" => DEFAULT_PY_STR.to_string(),
                                    "float" => DEFAULT_PY_FLOAT.to_string(),
                                    _ => DEFAULT_PY_NONE.to_string(),
                                },
                            )
                        });
                    format!("{}={}", arg_name, py_value)
                })
                .collect::<Vec<_>>()
                .join(", ");

        let (terminal_node_id, _edge_from_terminal) = path.last()?;
        let terminal_node = cfg.get_node(*terminal_node_id)?;

        let mut test_body_lines = Vec::new();
        let call_stmt = format!("{}({})", original_function_name, func_args_str);

        // Determine if the function's return type is float
        let is_return_type_float = cfg.get_fn_return_type() == Some(&"float".to_string());

        match terminal_node {
            Node::Return {
                stmts: _,
                stmt: return_stmt,
            } => {
                if let Some(expr_box) = &return_stmt.value {
                    match &expr_box.deref() {
                        // Use .deref() to get to Expr_
                        Expr::Constant(ExprConstant {
                            value: const_val, ..
                        }) => {
                            let expected_value_str = Self::format_python_constant(const_val);
                            if is_return_type_float && const_val.is_float() {
                                // Check if constant itself is float
                                test_body_lines.push(format!(
                                    "    assert {} == pytest.approx({})",
                                    call_stmt, expected_value_str
                                ));
                            } else {
                                test_body_lines.push(format!(
                                    "    assert {} == {}",
                                    call_stmt, expected_value_str
                                ));
                            }
                        }
                        Expr::Name(ExprName { id, .. }) => {
                            let returned_var_name = id.to_string();
                            let mut best_ssa_key: Option<String> = None;
                            let mut max_ssa_index: i32 = -1;

                            let ssa_prefix_expected_assigned =
                                format!("{}_assigned", returned_var_name);
                            let ssa_prefix_expected_aug_assigned =
                                format!("{}_aug_assigned", returned_var_name);

                            for (key_from_model, _value) in model_assignments.iter() {
                                if let Some(pos_bang) = key_from_model.rfind('!') {
                                    let prefix_in_key = &key_from_model[..pos_bang];
                                    let index_str = &key_from_model[(pos_bang + 1)..];

                                    let mut is_matching_ssa_pattern = false;
                                    if prefix_in_key == ssa_prefix_expected_assigned {
                                        is_matching_ssa_pattern = true;
                                    } else if prefix_in_key == ssa_prefix_expected_aug_assigned {
                                        is_matching_ssa_pattern = true;
                                    }

                                    if is_matching_ssa_pattern {
                                        if let Ok(current_index) = index_str.parse::<i32>() {
                                            if current_index > max_ssa_index {
                                                max_ssa_index = current_index;
                                                best_ssa_key = Some(key_from_model.clone());
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
                                        _ => model_value_str.clone(),
                                    };
                                    test_body_lines
                                        .push(format!("    returnValue = {}", call_stmt));
                                    if is_return_type_float {
                                        test_body_lines.push(format!(
                                            "    assert returnValue == pytest.approx({})",
                                            python_model_value
                                        ));
                                    } else {
                                        test_body_lines.push(format!(
                                            "    assert returnValue == {}",
                                            python_model_value
                                        ));
                                    }
                                    found_value_for_assertion = true;
                                }
                            }

                            if !found_value_for_assertion {
                                if let Some(model_value_str) =
                                    model_assignments.get(&returned_var_name)
                                {
                                    let python_model_value = match model_value_str.as_str() {
                                        "true" => "True".to_string(),
                                        "false" => "False".to_string(),
                                        _ => model_value_str.clone(),
                                    };
                                    test_body_lines
                                        .push(format!("    returnValue = {}", call_stmt));
                                    if is_return_type_float {
                                        test_body_lines.push(format!(
                                            "    assert returnValue == pytest.approx({})",
                                            python_model_value
                                        ));
                                    } else {
                                        test_body_lines.push(format!(
                                            "    assert returnValue == {}",
                                            python_model_value
                                        ));
                                    }
                                    found_value_for_assertion = true;
                                }
                            }

                            if !found_value_for_assertion {
                                test_body_lines.push(format!("    # Path returns variable '{}' whose value (or its SSA version) is not in the Z3 model for this path.", returned_var_name));
                                test_body_lines.push(format!("    returnValue = {}", call_stmt));
                                test_body_lines.push(
                                    "    # TODO: Add manual assertion for returnValue".to_string(),
                                );
                            }
                        }
                        Expr::UnaryOp(rustpython_ast::ExprUnaryOp { op, operand, .. }) => {
                            // Handle UnaryOp on a Constant, e.g., return -100.0
                            if let rustpython_ast::UnaryOp::USub = op {
                                if let Expr::Constant(ExprConstant {
                                    value: const_val, ..
                                }) = operand.deref()
                                {
                                    let mut expected_value_str =
                                        Self::format_python_constant(const_val);
                                    // Prepend negation, ensuring not to double-negate if format_python_constant already did (though unlikely for simple numbers)
                                    if !expected_value_str.starts_with('-')
                                        && expected_value_str != "0"
                                        && expected_value_str != "0.0"
                                    {
                                        expected_value_str = format!("-{}", expected_value_str);
                                    } else if expected_value_str.starts_with('-') {
                                        // It's already negative, e.g. from a complex number or a string representation
                                        // This case might need more nuanced handling if format_python_constant can produce negative strings for positive constants.
                                    }

                                    if is_return_type_float
                                        && (const_val.is_float() || const_val.is_int())
                                    {
                                        test_body_lines.push(format!(
                                            "    assert {} == pytest.approx({})",
                                            call_stmt, expected_value_str
                                        ));
                                    } else {
                                        test_body_lines.push(format!(
                                            "    assert {} == {}",
                                            call_stmt, expected_value_str
                                        ));
                                    }
                                } else {
                                    // UnaryOp on something other than a direct constant
                                    test_body_lines.push(format!(
                                        "    # Path returns a UnaryOp on a non-constant expression: {:?} applied to {:?}",
                                        op, operand.deref()
                                    ));
                                    test_body_lines
                                        .push(format!("    returnValue = {}", call_stmt));
                                    test_body_lines.push(
                                        "    # TODO: Add manual assertion for returnValue"
                                            .to_string(),
                                    );
                                }
                            } else {
                                // Other UnaryOps (Not, Invert, UAdd)
                                test_body_lines.push(format!(
                                    "    # Path returns an unhandled UnaryOp expression: {:?} applied to {:?}",
                                    op, operand.deref()
                                ));
                                test_body_lines.push(format!("    returnValue = {}", call_stmt));
                                test_body_lines.push(
                                    "    # TODO: Add manual assertion for returnValue".to_string(),
                                );
                            }
                        }
                        _ => {
                            test_body_lines.push(format!(
                                "    # Path returns a non-constant expression: {:?}",
                                expr_box.deref()
                            ));
                            test_body_lines.push(format!("    returnValue = {}", call_stmt));
                            test_body_lines.push(
                                "    # TODO: Add manual assertion for returnValue".to_string(),
                            );
                        }
                    }
                } else {
                    // Function returns None implicitly
                    test_body_lines.push(format!("    assert {} is None", call_stmt));
                }
            }
            Node::Raise {
                stmts: _,
                stmt: raise_stmt,
            } => {
                let exception_name = if let Some(exc_expr_box) = &raise_stmt.exc {
                    match &exc_expr_box.deref() {
                        // Use &exc_expr_box.node
                        Expr::Name(ExprName { id, .. }) => id.to_string(),
                        Expr::Call(ExprCall { func, .. }) => {
                            if let Expr::Name(ExprName { id, .. }) = &func.deref() {
                                // Use &func.node
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
                    test_body_lines.push(format!(
                        "    # Path raises a non-standard or complex exception: {:?}",
                        raise_stmt.exc.as_ref()
                    )); // Use .node
                } else if raise_stmt.exc.is_none() {
                    test_body_lines.push("    # Path involves a bare 'raise'".to_string());
                }

                test_body_lines.push(format!("    with pytest.raises({}):", exception_name));
                test_body_lines.push(format!("        {}", call_stmt));
            }
            Node::Cond { .. } => return None, // Should not happen as terminal node
        }

        if test_body_lines.is_empty() {
            return None;
        }

        let test_function_name = format!(
            "test_{}_path_{}",
            original_function_name
                .to_lowercase()
                .replace(|c: char| !c.is_alphanumeric() && c != '_', "_"),
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
