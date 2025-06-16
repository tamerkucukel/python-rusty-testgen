use crate::cfg::{ControlFlowGraph, Edge, Node, NodeId};
use crate::path::PathConstraintResult;
use rustpython_ast::{Constant, Expr, ExprCall, ExprConstant, ExprName, UnaryOp}; // Added UnaryOp
use std::collections::HashMap;
use std::fmt::Write;
use std::ops::Deref;

/// Converts Z3 real number S-expression strings to Python float literals.
///
/// Handles:
/// - Optional '?' suffix from Z3.
/// - Negation in the form "(- value)".
/// - Fractional form "(/ numerator denominator)".
/// - Attempts to parse and evaluate fractions to f64.
///
/// Note: This function preserves the original code's behavior, including how it
/// handles potentially malformed S-expressions or parsing failures, by returning
/// parts of the original S-expression string in such cases.
fn z3_real_to_python_float_literal(z3_val: &str) -> String {
    let mut s = z3_val.trim().to_string();
    if s.ends_with('?') {
        s.pop(); // Remove Z3's optional '?' suffix for reals
    }

    let original_s_for_fallback = s.clone(); // Used if parsing fails, to preserve original form

    let is_negative = s.starts_with("(- ") && s.ends_with(')');
    let mut current_value_str = if is_negative {
        // Preserves original potentially buggy behavior for malformed "(- value)"
        let stripped_prefix = s.strip_prefix("(- ").unwrap_or(&original_s_for_fallback);
        let stripped_suffix = stripped_prefix
            .strip_suffix(')')
            .unwrap_or(&original_s_for_fallback);
        stripped_suffix.trim().to_string()
    } else {
        s // Use the original string (already trimmed and '?' removed)
    };

    // Check for fractional form like "(/ numerator denominator)"
    if current_value_str.starts_with("(/ ") && current_value_str.ends_with(')') {
        let fraction_content = current_value_str
            .strip_prefix("(/ ")
            .unwrap_or("") // Fallback for safety, though prefix should exist
            .strip_suffix(')')
            .unwrap_or("") // Fallback for safety, though suffix should exist
            .trim();

        let mut parts_iter = fraction_content.split_whitespace();
        if let (Some(num_str_raw), Some(den_str_raw)) = (parts_iter.next(), parts_iter.next()) {
            // Recursively parse numerator and denominator, as they might be S-expressions
            let num_cleaned_py_literal = z3_real_to_python_float_literal(num_str_raw);
            let den_cleaned_py_literal = z3_real_to_python_float_literal(den_str_raw);

            if let (Ok(num_f64), Ok(den_f64)) = (
                num_cleaned_py_literal.parse::<f64>(),
                den_cleaned_py_literal.parse::<f64>(),
            ) {
                if den_f64 != 0.0 {
                    let result = num_f64 / den_f64;
                    let mut formatted_result = format!("{:.17}", result); // f64 has ~15-17 decimal digits precision
                    if formatted_result.contains('.') {
                        formatted_result = formatted_result.trim_end_matches('0').to_string();
                        if formatted_result.ends_with('.') {
                            formatted_result.push('0'); // Ensure "1." becomes "1.0"
                        }
                    }
                    current_value_str = formatted_result;
                } else {
                    // Division by zero: return original S-expression form (potentially negated)
                    return if is_negative {
                        format!("(- {})", current_value_str) // current_value_str is "(/ num den)" here
                    } else {
                        current_value_str // current_value_str is "(/ num den)" here
                    };
                }
            } else {
                // Failed to parse numerator or denominator as f64: return original S-expression
                return if is_negative {
                    format!("(- {})", current_value_str)
                } else {
                    current_value_str
                };
            }
        } else {
            // Malformed fraction string: return original S-expression
            return if is_negative {
                format!("(- {})", current_value_str)
            } else {
                current_value_str
            };
        }
    }
    // If not a fraction, current_value_str is now a cleaned decimal or integer string

    if is_negative {
        format!("-{}", current_value_str)
    } else {
        current_value_str
    }
}

/// A marker struct for generating Pytest test suites.
pub struct PytestGenerator;

// Default Python literal values for function arguments if not found in Z3 model.
const DEFAULT_PY_NONE: &str = "None";
const DEFAULT_PY_INT: &str = "0";
const DEFAULT_PY_BOOL: &str = "False";
const DEFAULT_PY_STR: &str = "\"\"";
const DEFAULT_PY_FLOAT: &str = "0.0";

/// Holds the generated import statements and test function strings for a single Python function.
pub struct GeneratedTestSuite {
    pub imports: Vec<String>,
    pub test_functions: Vec<String>,
}

impl PytestGenerator {
    /// Parses a Z3 model string into a map of variable names to their Python literal string values.
    ///
    /// Handles two Z3 model formats:
    /// 1. `(define-fun var_name () VarType value_expression)`
    /// 2. `var_name -> value_expression` (simpler format)
    ///
    /// Values are processed by `z3_real_to_python_float_literal` to convert Z3 Reals.
    pub(crate) fn parse_z3_model(model_str: &str) -> HashMap<String, String> {
        model_str
            .lines()
            .filter_map(|line| {
                let trimmed_line = line.trim();
                if trimmed_line.starts_with("(define-fun") && trimmed_line.ends_with(')') {
                    // Example: (define-fun x () Int 10)
                    // Example: (define-fun y () Real (- (/ 1.0 2.0)))
                    let core_content = trimmed_line
                        .strip_prefix("(define-fun ")?
                        .strip_suffix(')')?;
                    let parts: Vec<&str> = core_content.split_whitespace().collect();

                    if parts.len() >= 4 && parts[1] == "()" {
                        // parts[0] is name, parts[1] is "()", parts[2] is type
                        let name = parts[0].to_string();
                        let raw_value_s_expr = parts
                            .iter()
                            .skip(3)
                            .copied()
                            .collect::<Vec<&str>>()
                            .join(" ");
                        let python_literal_value =
                            z3_real_to_python_float_literal(&raw_value_s_expr);
                        Some((name, python_literal_value))
                    } else {
                        None
                    }
                } else if trimmed_line.contains("->") {
                    // Example: x -> 10
                    // Example: y -> (- 1.0)
                    let mut parts_iter = trimmed_line.splitn(2, "->");
                    if let (Some(name_str), Some(raw_value_str)) =
                        (parts_iter.next(), parts_iter.next())
                    {
                        let name = name_str.trim().to_string();
                        let python_literal_value =
                            z3_real_to_python_float_literal(raw_value_str.trim());
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

    /// Formats a `rustpython_ast::Constant` into its Python literal string representation.
    fn format_python_constant(constant: &Constant) -> String {
        match constant {
            Constant::Int(i) => i.to_string(),
            Constant::Bool(b) => if *b { "True" } else { "False" }.to_string(),
            Constant::Str(s_val) => {
                format!(
                    "\"{}\"",
                    s_val.as_str().replace('\\', "\\\\").replace('\"', "\\\"")
                )
            }
            Constant::Float(f) => f.to_string(),
            Constant::Complex { real, imag } => format!("complex({}, {})", real, imag),
            Constant::None => "None".to_string(),
            Constant::Ellipsis => "...".to_string(),
            Constant::Bytes(b_val) => {
                let mut repr = String::with_capacity(b_val.len() * 2 + 3); // Pre-allocate: b"" + escapes
                repr.push_str("b\"");
                for &byte_char in b_val {
                    match byte_char {
                        b'"' => repr.push_str("\\\""),
                        b'\\' => repr.push_str("\\\\"),
                        b'\n' => repr.push_str("\\n"),
                        b'\r' => repr.push_str("\\r"),
                        b'\t' => repr.push_str("\\t"),
                        32..=126 => repr.push(byte_char as char), // Printable ASCII
                        _ => write!(repr, "\\x{:02x}", byte_char)
                            .expect("Writing to String should not fail"),
                    }
                }
                repr.push('\"');
                repr
            }
            _ => {
                // For unsupported constants, return a placeholder string
                // This should not happen in well-formed ASTs, but is a fallback.
                format!("<unsupported constant: {:?}>", constant)
            }
        }
    }

    /// Generates the string for a single pytest test function based on a Z3 model and execution path.
    ///
    /// # Arguments
    /// * `original_function_name` - The name of the Python function being tested.
    /// * `path_index` - The index of the current path, used for unique test function naming.
    /// * `model_str` - The Z3 model string providing variable assignments for this path.
    /// * `path` - The sequence of (NodeId, Edge) representing the execution path.
    /// * `cfg` - The ControlFlowGraph of the function.
    ///
    /// # Returns
    /// An `Option<String>` containing the test function code, or `None` if generation fails
    /// (e.g., empty path, terminal node is not Return/Raise).
    #[allow(clippy::too_many_lines)] // This function is inherently complex due to AST traversal
    fn generate_test_function_string(
        original_function_name: &str,
        path_index: usize,
        model_str: &str,
        path_nodes_edges: &[(NodeId, Edge)],
        cfg: &ControlFlowGraph,
    ) -> Option<String> {
        let model_assignments = Self::parse_z3_model(model_str);

        let func_args_str =
            cfg.get_arguments()
                .iter()
                .map(|(arg_name, type_hint_opt)| {
                    let py_value = model_assignments
                        .get(arg_name)
                        .map(|model_value_str| match model_value_str.as_str() {
                            // Handle boolean string values from Z3 model explicitly
                            "true" => "True".to_string(),
                            "false" => "False".to_string(),
                            _ => model_value_str.clone(),
                        })
                        .unwrap_or_else(|| {
                            // Fallback to default values based on type hint
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

        let (terminal_node_id, _edge_from_terminal) = path_nodes_edges.last()?;
        let terminal_node = cfg.get_node(*terminal_node_id)?;

        let mut test_body_lines = Vec::new();
        let call_stmt = format!("{}({})", original_function_name, func_args_str);

        let is_return_type_float = cfg.get_fn_return_type().map_or(false, |rt| rt == "float");

        match terminal_node {
            Node::Return {
                stmt: return_stmt, ..
            } => {
                if let Some(expr_box) = &return_stmt.value {
                    match expr_box.deref() {
                        Expr::Constant(ExprConstant {
                            value: const_val, ..
                        }) => {
                            let expected_value_str = Self::format_python_constant(const_val);
                            // Use pytest.approx if function returns float AND constant is float
                            if is_return_type_float && const_val.is_float() {
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
                            let returned_var_name = id.as_str();
                            // Find the latest SSA version of the returned variable in the model
                            let latest_ssa_value = model_assignments
                                .iter()
                                .filter_map(|(key, value)| {
                                    key.strip_prefix(&format!("{}_assigned!", returned_var_name))
                                        .or_else(|| {
                                            key.strip_prefix(&format!(
                                                "{}_aug_assigned!",
                                                returned_var_name
                                            ))
                                        })
                                        .and_then(|index_str| index_str.parse::<i32>().ok())
                                        .map(|index| (index, value))
                                })
                                .max_by_key(|(index, _)| *index)
                                .map(|(_, value)| value)
                                .or_else(|| model_assignments.get(returned_var_name)); // Fallback to non-SSA name

                            if let Some(model_value_str) = latest_ssa_value {
                                let python_model_value = match model_value_str.as_str() {
                                    "true" => "True".to_string(),
                                    "false" => "False".to_string(),
                                    _ => model_value_str.clone(),
                                };
                                test_body_lines.push(format!("    returnValue = {}", call_stmt));
                                if is_return_type_float {
                                    // Approx for any var if func returns float
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
                            } else {
                                test_body_lines.push(format!("    # Path returns variable '{}' whose value (or its SSA version) is not in the Z3 model for this path.", returned_var_name));
                                test_body_lines.push(format!("    returnValue = {}", call_stmt));
                                test_body_lines.push(
                                    "    # TODO: Add manual assertion for returnValue".to_string(),
                                );
                            }
                        }
                        Expr::UnaryOp(unary_op_expr) => {
                            if let UnaryOp::USub = unary_op_expr.op {
                                if let Expr::Constant(ExprConstant {
                                    value: const_val, ..
                                }) = unary_op_expr.operand.deref()
                                {
                                    let mut expected_value_str =
                                        Self::format_python_constant(const_val);
                                    if !expected_value_str.starts_with('-')
                                        && expected_value_str != "0"
                                        && expected_value_str != "0.0"
                                    {
                                        expected_value_str = format!("-{}", expected_value_str);
                                    }
                                    // Use pytest.approx if func returns float AND constant was int/float
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
                                    test_body_lines.push(format!("    # Path returns a UnaryOp on a non-constant expression: {:?} applied to {:?}", unary_op_expr.op, unary_op_expr.operand.deref()));
                                    test_body_lines
                                        .push(format!("    returnValue = {}", call_stmt));
                                    test_body_lines.push(
                                        "    # TODO: Add manual assertion for returnValue"
                                            .to_string(),
                                    );
                                }
                            } else {
                                test_body_lines.push(format!("    # Path returns an unhandled UnaryOp expression: {:?} applied to {:?}", unary_op_expr.op, unary_op_expr.operand.deref()));
                                test_body_lines.push(format!("    returnValue = {}", call_stmt));
                                test_body_lines.push(
                                    "    # TODO: Add manual assertion for returnValue".to_string(),
                                );
                            }
                        }
                        _ => {
                            // Other complex expressions returned
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
                    // Implicit return None
                    test_body_lines.push(format!("    assert {} is None", call_stmt));
                }
            }
            Node::Raise {
                stmt: raise_stmt, ..
            } => {
                let exception_name = raise_stmt.exc.as_ref().map_or_else(
                    || "Exception".to_string(), // Bare raise or re-raise
                    |exc_expr| match exc_expr.deref() {
                        Expr::Name(ExprName { id, .. }) => id.to_string(),
                        Expr::Call(ExprCall { func, .. }) => {
                            if let Expr::Name(ExprName { id, .. }) = func.deref() {
                                id.to_string()
                            } else {
                                "Exception".to_string() // e.g. raise some_obj.Error()
                            }
                        }
                        _ => "Exception".to_string(), // e.g. raise "string" (invalid in Py3 but AST might allow)
                    },
                );

                if exception_name == "Exception" && raise_stmt.exc.is_some() {
                    test_body_lines.push(format!(
                        "# Path raises a non-standard or complex exception: {:?}",
                        raise_stmt.exc.as_ref()
                    ));
                } else if raise_stmt.exc.is_none() {
                    test_body_lines.push("# Path involves a bare 'raise' or re-raise".to_string());
                }

                test_body_lines.push(format!("    with pytest.raises({}):", exception_name));
                test_body_lines.push(format!("        {}", call_stmt));
            }
            Node::Cond { .. } => {
                // A Cond node should not be a terminal node in a valid complete path.
                // If it occurs, it implies an issue with path generation or CFG structure.
                // Return None to indicate no test function can be generated for this malformed path.
                return None;
            }
        }

        if test_body_lines.is_empty() {
            // This might happen if the terminal node is valid but doesn't lead to assertions
            // (e.g. a path ending in a simple pass statement, though CFGs usually end in Return/Raise).
            // Or if the terminal node was Cond, which now returns None above.
            return None;
        }

        let test_function_name = format!(
            "test_{}_path_{}",
            original_function_name
                .to_lowercase()
                .replace(|c: char| !c.is_alphanumeric() && c != '_', "_"),
            path_index
        );

        let mut fn_string_buffer = String::with_capacity(100 + test_body_lines.join("\n").len());
        // Using .ok()? ensures that if writeln! fails (highly unlikely for String),
        // the function correctly propagates None.
        writeln!(fn_string_buffer, "def {}():", test_function_name).ok()?;
        for line in test_body_lines {
            writeln!(fn_string_buffer, "{}", line).ok()?;
        }
        Some(fn_string_buffer)
    }

    /// Generates a test suite (imports and test functions) for a single Python function,
    /// based on the analysis results of its execution paths.
    ///
    /// # Arguments
    /// * `original_function_name` - The name of the Python function.
    /// * `path_results` - Slice of `PathConstraintResult` from Z3 analysis.
    /// * `all_paths` - Slice of all identified execution paths.
    /// * `cfg` - The `ControlFlowGraph` of the function.
    /// * `module_name_for_import` - Optional name of the module from which to import the function.
    ///
    /// # Returns
    /// A `GeneratedTestSuite` struct containing import strings and test function strings.
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
            // Provide a comment if the module name is unknown.
            imports.push(format!(
                "# Ensure '{}' is importable or defined in the test environment",
                original_function_name
            ));
        }
        imports.push("import pytest".to_string()); // Ensure pytest is imported for pytest.approx/raises

        let mut generated_any_test_for_this_func = false;

        for path_result in path_results.iter().filter(|pr| pr.is_satisfiable) {
            if let Some(model_str) = &path_result.model {
                // Ensure the path_index is valid for all_paths
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
            // Add a comment if paths were analyzed but no satisfiable paths led to test generation.
            // This helps distinguish from functions with no paths analyzed at all.
            test_functions.push(format!(
                "\n# No satisfiable paths led to test generation for function '{}'.",
                original_function_name
            ));
        } else if test_functions.is_empty() && path_results.is_empty() {
            // Comment if no paths were analyzed at all (e.g., CFG error before path finding)
            test_functions.push(format!(
                "\n# No paths were analyzed for function '{}', so no tests generated.",
                original_function_name
            ));
        }

        GeneratedTestSuite {
            imports,
            test_functions,
        }
    }
}
