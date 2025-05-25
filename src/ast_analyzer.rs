use std::ops::Deref;

use rustpython_ast::{
    Arg, ArgWithDefault, CmpOp, Constant, Expr, Stmt, StmtFor, StmtFunctionDef, StmtIf, StmtRaise,
    StmtReturn, StmtWhile, Visitor,
};
use rustpython_parser::Parse;




#[derive(Clone, Debug)]
/// Turns an expression and its sub-expressions into a string representation.
pub struct ExpressionParser;

impl ExpressionParser {
    /// Converts an exception type expression to its string representation.
    pub fn exception_type_to_str(&self, expr: &Expr) -> Option<String> {
        // Parse the expression to a string representation
        match expr {
            Expr::Name(name) => Some(name.id.to_string()), // Handles `raise Exception`
            Expr::Call(call) => {
                // Handles `raise Exception(...)`
                if let Expr::Name(name) = &*call.func {
                    Some(name.id.to_string())
                } else {
                    None
                }
            }
            _ => None, // Handle other expression types if needed
        }
    }

    /// Converts a comparison operator to its string representation.
    pub fn cmpop_to_str(&self, op: &CmpOp) -> Option<&'static str> {
        match op {
            CmpOp::Eq => Some("=="),
            CmpOp::NotEq => Some("!="),
            CmpOp::Lt => Some("<"),
            CmpOp::LtE => Some("<="),
            CmpOp::Gt => Some(">"),
            CmpOp::GtE => Some(">="),
            CmpOp::Is => Some("is"),
            CmpOp::IsNot => Some("is not"),
            CmpOp::In => Some("in"),
            CmpOp::NotIn => Some("not in"),
        }
    }
}

/// Extracts condition details from an if’s test expression.
///
/// For a Compare expression of the form "variable <operator> literal" or "literal <operator> variable",
/// returns (variable name, literal value string, operator string).
/// Limited to simple comparisons with a single operator and literal.
fn extract_condition_details(expr: &Expr) -> Option<(String, String, String)> {
    if let Expr::Compare(compare_expr) = expr {
        // Expect a single comparator and a single right-hand expression.
        if compare_expr.ops.len() == 1 && compare_expr.comparators.len() == 1 {
            let op = compare_expr.ops.get(0)?;
            let right_expr = compare_expr.comparators.get(0)?;

            // Case 1: variable <op> literal
            if let Expr::Name(var_name_expr) = &*compare_expr.left {
                if let Some(literal) = extract_literal_value(Some(right_expr)) {
                    let op_str = cmp_op_to_str(op)?;
                    return Some((var_name_expr.id.to_string(), literal, op_str.to_string()));
                }
            }
            // Case 2: literal <op> variable (need to flip operator for variable-centric view)
            if let Expr::Name(var_name_expr) = right_expr {
                if let Some(literal) = extract_literal_value(Some(&compare_expr.left)) {
                    // Flip the operator for logical equivalence based on variable
                    let flipped_op_str = match op {
                        CmpOp::Eq => Some("=="),
                        CmpOp::NotEq => Some("!="),
                        CmpOp::Lt => Some(">"),   // 10 < x is same as x > 10
                        CmpOp::LtE => Some(">="), // 10 <= x is same as x >= 10
                        CmpOp::Gt => Some("<"),   // 10 > x is same as x < 10
                        CmpOp::GtE => Some("<="), // 10 >= x is same as x <= 10
                        _ => None,                // Unsupported flipped operator
                    }?;
                    return Some((
                        var_name_expr.id.to_string(),
                        literal,
                        flipped_op_str.to_string(),
                    ));
                }
            }
        }
    }
    None // Condition is not a simple variable/literal comparison
}

/// Extracts the first Outcome (Return or Raise) from a list of statements.
///
/// It traverses the slice *linearly* and returns the first terminal statement found
/// at the top level of this slice. It does *not* recurse into nested control flow.
fn extract_outcome(stmts: &[Stmt]) -> Option<Outcome> {
    for stmt in stmts {
        match stmt {
            Stmt::Return(ret) => return Some(Outcome::Return(ret.clone())),
            Stmt::Raise(raise) => return Some(Outcome::Raise(raise.clone())),
            // Ignore other statements for outcome extraction at this level,
            // as they are not terminal.
            _ => continue,
        }
    }
    None // No terminal statement found at the top level of this block.
}

/// Extracts a literal value from an expression (if supported).
/// Returns a string representation suitable for test code.
fn extract_literal_value(expr_opt: Option<&Expr>) -> Option<String> {
    match expr_opt {
        Some(expr) => match expr {
            Expr::Constant(c) => {
                match &c.value {
                    Constant::None => Some("None".to_string()),
                    Constant::Bool(b) => Some(b.to_string()),
                    Constant::Str(s) => Some(format!("{:?}", s)), // Use debug fmt for quotes
                    Constant::Int(i) => Some(i.to_string()),
                    Constant::Float(f) => Some(f.to_string()),
                    // Add other supported constants if needed (Bytes, Tuple, etc.)
                    _ => None, // Unsupported constant type
                }
            }
            // In case the literal is given by a variable name (e.g. `if x == MY_CONSTANT`)
            // Treat the variable name as the literal for test generation purposes,
            // assuming it represents a constant value. This is a simplification.
            Expr::Name(n) => Some(n.id.to_string()),
            // Allow simple unary negation on constants like -1
            Expr::UnaryOp(unary_op) => {
                if let rustpython_ast::UnaryOp::USub = unary_op.op {
                    if let Expr::Constant(c) = &*unary_op.operand {
                        match &c.value {
                            Constant::Int(i) => Some(format!("-{}", i)),
                            Constant::Float(f) => Some(format!("-{}", f)),
                            _ => None,
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None, // Unsupported expression type for literal extraction
        },
        None => None,
    }
}

/// Collects FunctionMetrics by traversing the AST. Implements the Visitor trait.
#[derive(Clone, Debug)]
pub struct FunctionMetricCollector {
    pub function_metrics: Vec<FunctionMetric>,
    function_stack: Vec<FunctionMetric>, // Stack to handle nested functions
}

impl FunctionMetricCollector {
    pub fn new() -> Self {
        Self {
            function_metrics: Vec::new(),
            function_stack: Vec::new(),
        }
    }

    // Private helper to get mutable reference to current function metric
    fn current_function_mut(&mut self) -> Option<&mut FunctionMetric> {
        self.function_stack.last_mut()
    }
}

// Visitor trait implementation.
impl Visitor for FunctionMetricCollector {
    fn visit_stmt_function_def(&mut self, node: StmtFunctionDef) {
        // Start a new function metric.
        let mut func_metric = FunctionMetric::new();
        // Clone the name (String) and arguments (Vec<Arg>) explicitly for storage
        func_metric.name = node.name.to_string();
        func_metric.arguments = node.args.args.iter().map(|arg| arg.def.clone()).collect(); // Clone ArgDef

        self.function_stack.push(func_metric);

        // Visit the function body, arguments, etc.
        // generic_visit_* takes a reference to the node.
        self.generic_visit_stmt_function_def(node.clone());

        // After visiting the function, record its metric.
        // The stack should not be empty here if we entered a function definition.
        if let Some(completed) = self.function_stack.pop() {
            self.function_metrics.push(completed);
        } else {
            // This indicates an issue in stack management if it's empty here
            eprintln!(
                "Warning: Function stack was empty after visiting function '{}'",
                node.name
            );
        }
    }

    fn visit_stmt_if(&mut self, node: StmtIf) {
        // Try to extract decision details from the if condition using the standalone helper.
        if let Some((var_name, literal, op)) = extract_condition_details(&node.test) {
            // Use the standalone helper to extract outcomes from the blocks.
            // extract_outcome needs &[Stmt], so pass references to the vecs.
            let then_outcome = extract_outcome(&node.body);
            let else_outcome = if !node.orelse.is_empty() {
                extract_outcome(&node.orelse)
            } else {
                None
            };

            // Only record the decision point if we captured at least one outcome
            if then_outcome.is_some() || else_outcome.is_some() {
                let dp = DecisionPoint {
                    var_name,
                    literal,
                    op,
                    then_outcome, // These are clones from extract_outcome
                    else_outcome, // These are clones from extract_outcome
                };
                if let Some(current) = self.current_function_mut() {
                    current.decision_points.push(dp);
                }
            } else {
                // Optionally log skipped IFs if desired
                // eprintln!("Skipping decision point analysis for IF: No top-level terminal outcome found in body or orelse.");
            }
        } else {
            // Optionally log skipped complex IFs if desired
            // eprintln!("Skipping decision point analysis for IF: Condition was not a simple variable/literal comparison.");
        }

        // Visit children of the if statement (body and orelse).
        self.generic_visit_stmt_if(node);
    }

    fn visit_stmt_return(&mut self, node: StmtReturn) {
        if let Some(current) = self.current_function_mut() {
            // Add all returns found *within* a function scope.
            current.return_defs.push(node.clone()); // Clone to store the node
        }
        // Visit children of the return statement (e.g., the return value expression)
        self.generic_visit_stmt_return(node);
    }

    fn visit_stmt_raise(&mut self, node: StmtRaise) {
        if let Some(current) = self.current_function_mut() {
            // Add all raises found *within* a function scope.
            current.raise_defs.push(node.clone()); // Clone to store the node
        }
        // Visit children of the raise statement (e.g., the exception expression)
        self.generic_visit_stmt_raise(node);
    }

    // Ensure other visit methods also take references if implementing specific logic,
    // or rely on the default generic traversal which now uses references.
    fn visit_stmt_while(&mut self, node: StmtWhile) {
        self.generic_visit_stmt_while(node);
    }

    fn visit_stmt_for(&mut self, node: StmtFor) {
        self.generic_visit_stmt_for(node);
    }

    // The visit_stmt method is crucial for traversing all statements within a block.
    // It typically calls generic_visit_stmt which dispatches to the specific visit_stmt_* methods.
    fn visit_stmt(&mut self, node: Stmt) {
        self.generic_visit_stmt(node);
    }

    // Add visit methods for expressions, etc., if needed for deeper traversal beyond generic.
    // Example:
    // fn visit_expr(&mut self, node: &Expr) { self.generic_visit_expr(node); }
}

// --- Test Generation Logic (Standalone Function) ---

/// Generates test code covering simple decision branches and default outcomes.
///
/// For each simple decision point (if variable <op> literal) it generates two tests (if an outcome exists and is analyzable for that branch):
/// - A test for the branch when the if condition is true.
/// - A test for the branch when the if condition is false.
///
/// Additionally, if the function contains top-level returns/raises and no simple decision points were extracted,
/// it generates a default test for the last encountered return/raise if analyzable.
pub fn generate_tests_for_function(function: &FunctionMetric) -> String {
    let mut test_funcs = Vec::new();
    let func_name = &function.name;

    // Helper: produce a default argument list string.
    // For simplicity, we assume numeric defaults of "0" for parameters unless overridden.
    // String defaults are empty strings "". This is a heuristic.
    fn default_args_string(args: &[Arg], override_name: Option<(&str, &str)>) -> String {
        args.iter()
            .map(|arg| {
                let arg_name = arg.arg.as_str();
                if let Some((name_to_override, override_value)) = override_name {
                    if arg_name == name_to_override {
                        return override_value.to_string();
                    }
                }
                // Simple heuristic based on common types or lack of annotation
                // Inspect annotation first if available
                if let Some(annotation_expr) = arg.annotation.as_deref() {
                    if let Expr::Name(name_expr) = annotation_expr {
                        match name_expr.id.as_str() {
                            "str" => "\"\"".to_string(),
                            "int" | "float" | "complex" => "0".to_string(),
                            "bool" => "False".to_string(),
                            "list" | "tuple" | "dict" | "set" => "None".to_string(), // Use None or appropriate empty literal if possible
                            _ => "0".to_string(), // Default for unhandled types
                        }
                    } else {
                        // Annotation is not a simple Name (e.g., List[int]) - fallback
                        "0".to_string()
                    }
                } else {
                    "0".to_string() // Default if no annotation
                }
            })
            .collect::<Vec<String>>()
            .join(", ")
    }

    // Helper: generate a test value based on the literal, comparator operator, and desired branch.
    // branch should be "then" (to satisfy the condition) or "else" (to not satisfy it).
    // This is a heuristic and may not work for all types/operators/literals.
    fn generate_test_value(literal: &str, op: &str, branch: &str) -> String {
        // Attempt to parse as various types
        if let Ok(n) = literal.parse::<i64>() {
            match op {
                "==" => {
                    if branch == "then" {
                        n.to_string()
                    } else {
                        (n + 1).to_string()
                    }
                }
                "!=" => {
                    if branch == "then" {
                        (n + 1).to_string()
                    } else {
                        n.to_string()
                    }
                }
                "<" => {
                    if branch == "then" {
                        (n - 1).to_string()
                    } else {
                        n.to_string()
                    }
                }
                "<=" => {
                    if branch == "then" {
                        n.to_string()
                    } else {
                        (n + 1).to_string()
                    }
                }
                ">" => {
                    if branch == "then" {
                        (n + 1).to_string()
                    } else {
                        n.to_string()
                    }
                }
                ">=" => {
                    if branch == "then" {
                        n.to_string()
                    } else {
                        (n - 1).to_string()
                    }
                }
                _ => literal.to_string(), // Fallback
            }
        } else if let Ok(f) = literal.parse::<f64>() {
            match op {
                "==" => {
                    if branch == "then" {
                        f.to_string()
                    } else {
                        (f + 1.0).to_string()
                    }
                }
                "!=" => {
                    if branch == "then" {
                        (f + 1.0).to_string()
                    } else {
                        f.to_string()
                    }
                }
                "<" => {
                    if branch == "then" {
                        (f - 1.0).to_string()
                    } else {
                        f.to_string()
                    }
                }
                "<=" => {
                    if branch == "then" {
                        f.to_string()
                    } else {
                        (f + 1.0).to_string()
                    }
                }
                ">" => {
                    if branch == "then" {
                        (f + 1.0).to_string()
                    } else {
                        f.to_string()
                    }
                }
                ">=" => {
                    if branch == "then" {
                        f.to_string()
                    } else {
                        (f - 1.0).to_string()
                    }
                }
                _ => literal.to_string(), // Fallback
            }
        } else if literal.starts_with('\'') || literal.starts_with('"') {
            // It's a string literal representation (e.g., '"abc"'), strip quotes
            let clean_str =
                if literal.len() >= 2 && literal.starts_with('\'') && literal.ends_with('\'') {
                    &literal[1..literal.len() - 1]
                } else if literal.len() >= 2 && literal.starts_with('"') && literal.ends_with('"') {
                    &literal[1..literal.len() - 1]
                } else {
                    literal // Should not happen if it started with quote
                };

            let new_val = match op {
                "==" => {
                    if branch == "then" {
                        clean_str.to_string()
                    } else {
                        format!("{}_alt", clean_str)
                    }
                }
                "!=" => {
                    if branch == "then" {
                        format!("{}_alt", clean_str)
                    } else {
                        clean_str.to_string()
                    }
                }
                // Simple heuristic for string ordering comparisons
                "<" | "<=" => {
                    if branch == "then" {
                        format!("a{}", clean_str)
                    } else {
                        clean_str.to_string()
                    }
                }
                ">" | ">=" => {
                    if branch == "then" {
                        format!("z{}", clean_str)
                    } else {
                        clean_str.to_string()
                    }
                }
                _ => clean_str.to_string(), // Fallback
            };
            // Restore quotes using debug format for reliability
            format!("{:?}", new_val)
        } else if literal == "True" {
            match op {
                "==" => {
                    if branch == "then" {
                        "True".to_string()
                    } else {
                        "False".to_string()
                    }
                }
                "!=" => {
                    if branch == "then" {
                        "False".to_string()
                    } else {
                        "True".to_string()
                    }
                }
                _ => literal.to_string(), // Ordering not applicable
            }
        } else if literal == "False" {
            match op {
                "==" => {
                    if branch == "then" {
                        "False".to_string()
                    } else {
                        "True".to_string()
                    }
                }
                "!=" => {
                    if branch == "then" {
                        "True".to_string()
                    } else {
                        "False".to_string()
                    }
                }
                _ => literal.to_string(), // Ordering not applicable
            }
        } else if literal == "None" {
            match op {
                "==" => {
                    if branch == "then" {
                        "None".to_string()
                    } else {
                        "1".to_string()
                    }
                } // Use a non-None value
                "!=" => {
                    if branch == "then" {
                        "1".to_string()
                    } else {
                        "None".to_string()
                    }
                } // Use a non-None value
                _ => literal.to_string(), // Ordering not applicable
            }
        }
        // Add handling for other literal types if needed (Bytes, Tuple, etc.)
        else {
            // Fallback for potentially unhandled literals (like variable names treated as literals)
            literal.to_string()
        }
    }

    // Helper to format outcome into assertion code
    fn format_outcome_assertion(
        outcome: &Outcome,
        func_name: &str,
        args_string: &str,
    ) -> Option<String> {
        match outcome {
            Outcome::Return(ret_stmt) => {
                // Use the standalone helper for extracting return values
                if let Some(expected) = extract_literal_value(ret_stmt.value.as_deref()) {
                    Some(format!(
                        "    assert {}({}) == {}",
                        func_name, args_string, expected
                    ))
                } else {
                    // Cannot generate assertion for non-literal return
                    None
                }
            }
            Outcome::Raise(raise_stmt) => {
                // Use the standalone helper for extracting exception types
                if let Some(exc_type) = extract_exception_type(raise_stmt.exc.as_deref()) {
                    Some(format!(
                        "    with pytest.raises({}):\n        {}({})",
                        exc_type, func_name, args_string
                    ))
                } else {
                    // Cannot generate assertion for non-simple raise
                    None
                }
            }
        }
    }

    // Generate tests for each simple decision point found.
    for (i, dp) in function.decision_points.iter().enumerate() {
        let var = &dp.var_name;
        let op = &dp.op;
        let lit = &dp.literal;

        // Generate test for the then branch if an outcome exists and is analyzable.
        if let Some(ref outcome) = dp.then_outcome {
            let then_val = generate_test_value(lit, op, "then");
            let arg_assignment = default_args_string(&function.arguments, Some((var, &then_val)));

            if let Some(assertion_code) =
                format_outcome_assertion(outcome, func_name, &arg_assignment)
            {
                let test_code = format!(
                    r#"def test_{func_name}_dp{i}_then():
    # When {var} {op} {lit} holds (using {val}), expect outcome.
{assertion_code}
"#,
                    func_name = func_name,
                    i = i,
                    var = var,
                    op = op,
                    lit = lit,
                    val = then_val,
                    assertion_code = assertion_code
                );
                test_funcs.push(test_code);
            } else {
                eprintln!("Warning: Could not generate test for then branch outcome of '{func_name}' decision point {i} (variable: {var}, literal: {lit}, op: {op}). Outcome type unsupported or complex.");
            }
        }

        // Generate test for the else branch if an outcome exists and is analyzable.
        if let Some(ref outcome) = dp.else_outcome {
            let else_val = generate_test_value(lit, op, "else");
            let arg_assignment = default_args_string(&function.arguments, Some((var, &else_val)));

            if let Some(assertion_code) =
                format_outcome_assertion(outcome, func_name, &arg_assignment)
            {
                let test_code = format!(
                    r#"def test_{func_name}_dp{i}_else():
    # When {var} {op} {lit} does not hold (using {val}), expect outcome.
{assertion_code}
"#,
                    func_name = func_name,
                    i = i,
                    var = var,
                    op = op,
                    lit = lit,
                    val = else_val,
                    assertion_code = assertion_code
                );
                test_funcs.push(test_code);
            } else {
                eprintln!("Warning: Could not generate test for else branch outcome of '{func_name}' decision point {i} (variable: {var}, literal: {lit}, op: {op}). Outcome type unsupported or complex.");
            }
        }
    }

    // If no simple decision points were found, attempt to generate a default test
    // based on the last captured top-level return or raise.
    if function.decision_points.is_empty() {
        let arg_assignment = default_args_string(&function.arguments, None);
        let mut default_test_generated = false;

        if let Some(last_return) = function.return_defs.last() {
            if let Some(assertion_code) = format_outcome_assertion(
                &Outcome::Return(last_return.clone()),
                func_name,
                &arg_assignment,
            ) {
                let test_code = format!(
                    r#"def test_{func_name}_default():
    # Default test for function with no simple decision points, based on last return.
{assertion_code}
"#,
                    func_name = func_name,
                    assertion_code = assertion_code
                );
                test_funcs.push(test_code);
                default_test_generated = true;
            }
        } else if let Some(last_raise) = function.raise_defs.last() {
            if let Some(assertion_code) = format_outcome_assertion(
                &Outcome::Raise(last_raise.clone()),
                func_name,
                &arg_assignment,
            ) {
                // Import pytest needed for pytest.raises
                let test_code = format!(
                    r#"import pytest

def test_{func_name}_default_exception():
    # Default test for function with no simple decision points, based on last raise.
{assertion_code}
"#,
                    func_name = func_name,
                    assertion_code = assertion_code
                );
                test_funcs.push(test_code);
                default_test_generated = true;
            }
        }

        if !default_test_generated
            && (!function.return_defs.is_empty() || !function.raise_defs.is_empty())
        {
            eprintln!("Warning: Could not generate a default test for function '{func_name}' based on its final return/raise. Outcome type unsupported or complex.");
        } else if function.decision_points.is_empty()
            && function.return_defs.is_empty()
            && function.raise_defs.is_empty()
        {
            eprintln!("Info: No simple decision points or top-level terminal outcomes found in '{func_name}'. No tests generated.");
        }
    }

    if test_funcs.is_empty() {
        // Add a comment explaining why no tests were generated if the list is still empty
        format!("# No simple decision points or analyzable terminal outcomes found for function '{}'\n# No tests generated.", func_name)
    } else {
        // Add pytest import at the top if any test uses pytest.raises
        let needs_pytest_import = test_funcs.iter().any(|test| test.contains("pytest.raises"));
        let mut final_output = if needs_pytest_import {
            "import pytest\n\n".to_string()
        } else {
            String::new()
        };
        final_output.push_str(&test_funcs.join("\n\n"));
        final_output
    }
}

// --- Top-level Orchestration ---

/// Parses Python code, analyzes functions using the collector, and returns the collected metrics.
/// Returns a Result indicating success or a parsing error.
pub fn analyze_python_code(
    code: &str,
) -> Result<Vec<FunctionMetric>, rustpython_parser::ParseError> {
    let ast: Vec<Stmt> = Parse::parse_without_path(code)?;

    let mut collector = FunctionMetricCollector::new();

    // Visit each top-level statement by reference.
    for stmt in &ast {
        // Iterate over references
        collector.visit_stmt(stmt.clone()); // Pass reference to visit_stmt
    }

    Ok(collector.function_metrics)
}
