use rustpython_ast::{
    Arg, Arguments, CmpOp, Expr, Stmt, StmtFor, StmtFunctionDef, StmtIf, StmtRaise, StmtReturn,
    StmtWhile, Visitor,
};
use rustpython_parser::Parse;

/// Represents a function outcome along one execution path.
#[derive(Clone, Debug)]
pub enum Outcome {
    Return(StmtReturn),
    Raise(StmtRaise),
}

/// Represents an if-statement decision point.
#[derive(Clone, Debug)]
pub struct DecisionPoint {
    pub var_name: String, // the function argument being compared
    pub literal: String,  // literal value involved in the comparison
    pub op: String,       // comparator operator as a string (e.g., "==", "<", etc.)
    pub then_outcome: Option<Outcome>,
    pub else_outcome: Option<Outcome>,
}

/// Holds metrics for a function.
#[derive(Clone, Debug)]
pub struct FunctionMetric {
    pub name: String,
    pub arguments: Vec<Arg>,
    pub return_defs: Vec<StmtReturn>,
    pub raise_defs: Vec<StmtRaise>,
    // Holds the decision points captured from if statements.
    pub decision_points: Vec<DecisionPoint>,
}

impl FunctionMetric {
    pub fn new() -> Self {
        Self {
            name: String::new(),
            arguments: Vec::new(),
            return_defs: Vec::new(),
            raise_defs: Vec::new(),
            decision_points: Vec::new(),
        }
    }
}

/// The ASTAnalyzer now uses a stack to correctly isolate metrics for nested functions.
#[derive(Clone, Debug)]
pub struct ASTAnalyzer {
    pub function_metrics: Vec<FunctionMetric>,
    pub function_stack: Vec<FunctionMetric>,
}

impl ASTAnalyzer {
    pub fn new() -> Self {
        Self {
            function_metrics: Vec::new(),
            function_stack: Vec::new(),
        }
    }

    /// Generates the AST from the provided Python code.
    pub fn generate_ast(&self, code: &str) -> Vec<Stmt> {
        Parse::parse_without_path(code).expect("Couldn't parse python code")
    }

    /// Entry point for AST traversal.
    pub fn visit(&mut self, nodes: &[Stmt]) {
        for node in nodes {
            self.visit_stmt(node.clone());
        }
    }

    /// Generates test code covering each decision path found in a function.
    ///
    /// For each decision point (if statement) it generates two tests (if an outcome exists for that branch):
    /// - A test for the branch when the if condition is true.
    /// - A test for the branch when the if condition is false.
    ///
    /// Additionally, if the function contains a final return (outside of any decision) and no decision points,
    /// it generates a default success test.
    pub fn generate_test(function: &FunctionMetric) -> String {
        let mut test_funcs = Vec::new();
        let func_name = &function.name;

        // Helper: produce a default argument list.
        // For simplicity, we assume numeric defaults of "0" for parameters.
        // The 'override_name' allows us to override one argument's value.
        fn default_args(args: &[Arg], override_name: Option<(&str, &str)>) -> String {
            args.iter()
                .map(|arg| {
                    if let Some((name, value)) = override_name {
                        if arg.arg.to_string() == name {
                            return value.to_string();
                        }
                    }
                    "0".to_string()
                })
                .collect::<Vec<String>>()
                .join(", ")
        }

        // Helper: generate a test value based on the literal, comparator operator, and desired branch.
        // branch should be "then" (to satisfy the condition) or "else" (to not satisfy it).
        fn generate_test_value(literal: &str, op: &str, branch: &str) -> String {
            // Try numeric first.
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
                            (n + 1).to_string() // a value different from n
                        } else {
                            n.to_string()
                        }
                    }
                    "<" => {
                        if branch == "then" {
                            (n - 1).to_string() // must be less than n
                        } else {
                            n.to_string() // equals n, and n < n is false
                        }
                    }
                    "<=" => {
                        if branch == "then" {
                            n.to_string() // n <= n is true
                        } else {
                            (n + 1).to_string() // greater than n
                        }
                    }
                    ">" => {
                        if branch == "then" {
                            (n + 1).to_string() // greater than n
                        } else {
                            n.to_string() // n > n is false
                        }
                    }
                    ">=" => {
                        if branch == "then" {
                            n.to_string() // n >= n is true
                        } else {
                            (n - 1).to_string() // less than n
                        }
                    }
                    _ => n.to_string(), // fallback to literal itself
                }
            } else {
                // Handle non-numeric (assume string literal).
                // Strip quotes if present.
                let stripped = if literal.starts_with('\"') && literal.ends_with('\"') {
                    &literal[1..literal.len() - 1]
                } else {
                    literal
                };
                let new_val = match op {
                    "==" => {
                        if branch == "then" {
                            stripped.to_string()
                        } else {
                            format!("{}_alt", stripped)
                        }
                    }
                    "!=" => {
                        if branch == "then" {
                            format!("{}_alt", stripped)
                        } else {
                            stripped.to_string()
                        }
                    }
                    // For ordering, we simply append a suffix to try to flip lexicographic order.
                    "<" | "<=" => {
                        if branch == "then" {
                            // For then branch, return a value that sorts before the literal.
                            format!("{}a", stripped)
                        } else {
                            // Else branch: return the original literal.
                            stripped.to_string()
                        }
                    }
                    ">" | ">=" => {
                        if branch == "then" {
                            // For then branch, return a value that sorts after the literal.
                            format!("{}z", stripped)
                        } else {
                            stripped.to_string()
                        }
                    }
                    _ => stripped.to_string(),
                };
                // Re-add the quotes.
                format!("\"{}\"", new_val)
            }
        }

        // Generate tests for each decision point.
        for (i, dp) in function.decision_points.iter().enumerate() {
            // Generate test for the then branch if an outcome exists.
            if let Some(ref outcome) = dp.then_outcome {
                let then_val = generate_test_value(&dp.literal, &dp.op, "then");
                let arg_assignment =
                    default_args(&function.arguments, Some((&dp.var_name, &then_val)));
                match outcome {
                    Outcome::Return(ret_stmt) => {
                        if let Some(expected) =
                            Self::extract_literal_value(ret_stmt.value.as_deref())
                        {
                            let test_code = format!(
                                r#"def test_{func_name}_dp{i}_then():
    # When {var} {op} {lit} holds (using {val}), expect a literal return value.
    assert {func_name}({args}) == {expected}
"#,
                                func_name = func_name,
                                i = i,
                                var = dp.var_name,
                                op = dp.op,
                                lit = dp.literal,
                                val = then_val,
                                args = arg_assignment,
                                expected = expected
                            );
                            test_funcs.push(test_code);
                        }
                    }
                    Outcome::Raise(raise_stmt) => {
                        if let Some(exc_type) =
                            Self::extract_exception_type(raise_stmt.exc.as_deref())
                        {
                            let test_code = format!(
                                r#"import pytest

def test_{func_name}_dp{i}_then_exception():
    # When {var} {op} {lit} holds (using {val}), expect an exception.
    with pytest.raises({exc}):
        {func_name}({args})
"#,
                                func_name = func_name,
                                i = i,
                                var = dp.var_name,
                                op = dp.op,
                                lit = dp.literal,
                                val = then_val,
                                exc = exc_type,
                                args = arg_assignment
                            );
                            test_funcs.push(test_code);
                        }
                    }
                }
            }

            // Generate test for the else branch if an outcome exists.
            if let Some(ref outcome) = dp.else_outcome {
                let else_val = generate_test_value(&dp.literal, &dp.op, "else");
                let arg_assignment =
                    default_args(&function.arguments, Some((&dp.var_name, &else_val)));
                match outcome {
                    Outcome::Return(ret_stmt) => {
                        if let Some(expected) =
                            Self::extract_literal_value(ret_stmt.value.as_deref())
                        {
                            let test_code = format!(
                                r#"def test_{func_name}_dp{i}_else():
    # When {var} {op} {lit} does not hold (using {val}), expect a literal return value.
    assert {func_name}({args}) == {expected}
"#,
                                func_name = func_name,
                                i = i,
                                var = dp.var_name,
                                op = dp.op,
                                lit = dp.literal,
                                val = else_val,
                                args = arg_assignment,
                                expected = expected
                            );
                            test_funcs.push(test_code);
                        }
                    }
                    Outcome::Raise(raise_stmt) => {
                        if let Some(exc_type) =
                            Self::extract_exception_type(raise_stmt.exc.as_deref())
                        {
                            let test_code = format!(
                                r#"import pytest

def test_{func_name}_dp{i}_else_exception():
    # When {var} {op} {lit} does not hold (using {val}), expect an exception.
    with pytest.raises({exc}):
        {func_name}({args})
"#,
                                func_name = func_name,
                                i = i,
                                var = dp.var_name,
                                op = dp.op,
                                lit = dp.literal,
                                val = else_val,
                                exc = exc_type,
                                args = arg_assignment
                            );
                            test_funcs.push(test_code);
                        }
                    }
                }
            }
        }

        // If no decision points were found, fall back to a simple test based on the final return outcome.
        if function.decision_points.is_empty() && !function.return_defs.is_empty() {
            if let Some(last_return) = function.return_defs.last() {
                if let Some(expected) = Self::extract_literal_value(last_return.value.as_deref()) {
                    let arg_assignment = default_args(&function.arguments, None);
                    let test_code = format!(
                        r#"def test_{func_name}_default():
    assert {func_name}({args}) == {expected}
"#,
                        func_name = func_name,
                        args = arg_assignment,
                        expected = expected
                    );
                    test_funcs.push(test_code);
                }
            }
        }

        test_funcs.join("\n")
    }

    /// Extracts a literal value from an expression (if supported).
    fn extract_literal_value(expr_opt: Option<&Expr>) -> Option<String> {
        match expr_opt {
            Some(expr) => match expr {
                // Support constant values.
                Expr::Constant(c) => Some(format!("{:?}", c.value)),
                // In case the literal is given by a variable name.
                Expr::Name(n) => Some(n.id.to_string()),
                _ => None,
            },
            None => None,
        }
    }

    /// Extracts the exception type from an optional expression in a raise statement.
    fn extract_exception_type(expr_opt: Option<&Expr>) -> Option<String> {
        match expr_opt {
            Some(expr) => match expr {
                Expr::Call(call_expr) => {
                    if let Expr::Name(ref name_expr) = *call_expr.func {
                        Some(name_expr.id.to_string())
                    } else {
                        None
                    }
                }
                Expr::Name(name_expr) => Some(name_expr.id.to_string()),
                _ => None,
            },
            None => None,
        }
    }

    /// Extracts condition details from an if’s test expression.
    ///
    /// For a Compare expression of the form "variable <operator> literal", returns (variable, literal, operator).
    fn extract_condition_details(expr: &Expr) -> Option<(String, String, String)> {
        if let Expr::Compare(compare_expr) = expr {
            // Check that the left side is a Name.
            if let Expr::Name(var) = &*compare_expr.left {
                // Expect a single comparator.
                if compare_expr.ops.len() == 1 && compare_expr.comparators.len() == 1 {
                    if let Some(comp_expr) = compare_expr.comparators.get(0) {
                        if let Some(literal) = Self::extract_literal_value(Some(comp_expr)) {
                            // Map the comparator to a string.
                            let op_str = match compare_expr.ops.get(0)? {
                                CmpOp::Eq => "==",
                                CmpOp::NotEq => "!=",
                                CmpOp::Lt => "<",
                                CmpOp::LtE => "<=",
                                CmpOp::Gt => ">",
                                CmpOp::GtE => ">=",
                                _ => return None, // unsupported operator
                            };
                            return Some((var.id.to_string(), literal, op_str.to_string()));
                        }
                    }
                }
            }
        }
        None
    }

    /// Extracts the first Outcome (Return or Raise) from a list of statements.
    ///
    /// It traverses the slice until it finds a return or raise.
    fn extract_outcome(stmts: &[Stmt]) -> Option<Outcome> {
        for stmt in stmts {
            match stmt {
                Stmt::Return(ret) => return Some(Outcome::Return(ret.clone())),
                Stmt::Raise(raise) => return Some(Outcome::Raise(raise.clone())),
                // For nested if-statements, traverse deeper if required.
                Stmt::If(if_stmt) => {
                    if let Some(outcome) = Self::extract_outcome(&if_stmt.body) {
                        return Some(outcome);
                    } else if !if_stmt.orelse.is_empty() {
                        if let Some(outcome) = Self::extract_outcome(&if_stmt.orelse) {
                            return Some(outcome);
                        }
                    }
                }
                _ => continue,
            }
        }
        None
    }
}

impl Visitor for ASTAnalyzer {
    fn visit_stmt_function_def(&mut self, node: StmtFunctionDef) {
        // Start a new function metric.
        let mut func_metric = FunctionMetric::new();
        func_metric.name = node.name.to_string();
        self.function_stack.push(func_metric);

        self.generic_visit_stmt_function_def(node);

        // After visiting the function, record its metric.
        if let Some(completed) = self.function_stack.pop() {
            self.function_metrics.push(completed);
        }
    }

    fn visit_arguments(&mut self, node: Arguments) {
        if let Some(current) = self.function_stack.last_mut() {
            for arg in &node.args {
                current.arguments.push(arg.def.clone());
            }
        }
        self.generic_visit_arguments(node);
    }

    fn visit_stmt_if(&mut self, node: StmtIf) {
        // Try to extract decision details from the if condition.
        if let Some((var_name, literal, op)) = Self::extract_condition_details(&node.test) {
            let then_outcome = Self::extract_outcome(&node.body);
            let else_outcome = if !node.orelse.is_empty() {
                Self::extract_outcome(&node.orelse)
            } else {
                None
            };

            let dp = DecisionPoint {
                var_name,
                literal,
                op,
                then_outcome,
                else_outcome,
            };
            if let Some(current) = self.function_stack.last_mut() {
                current.decision_points.push(dp);
            }
        }
        self.generic_visit_stmt_if(node);
    }

    fn visit_stmt_return(&mut self, node: StmtReturn) {
        if let Some(current) = self.function_stack.last_mut() {
            current.return_defs.push(node.clone());
        }
        self.generic_visit_stmt_return(node);
    }

    fn visit_stmt_raise(&mut self, node: StmtRaise) {
        if let Some(current) = self.function_stack.last_mut() {
            current.raise_defs.push(node.clone());
        }
        self.generic_visit_stmt_raise(node);
    }

    fn visit_stmt_while(&mut self, node: StmtWhile) {
        self.generic_visit_stmt_while(node);
    }

    fn visit_stmt_for(&mut self, node: StmtFor) {
        self.generic_visit_stmt_for(node);
    }
}
