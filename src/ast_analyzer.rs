use rustpython_ast::{
    Arg, Arguments, Expr, Stmt, StmtFor, StmtFunctionDef, StmtIf, StmtRaise, StmtReturn, StmtWhile,
    Visitor,
};
use rustpython_parser::Parse;

// Stores metrics for each function in terms of decision points and outcomes.
#[derive(Clone, Debug)]
pub struct FunctionMetric {
    pub name: String,
    pub arguments: Vec<Arg>,
    pub if_defs: Vec<StmtIf>,
    pub while_defs: Vec<StmtWhile>,
    pub for_defs: Vec<StmtFor>,
    pub return_defs: Vec<StmtReturn>,
    pub raise_defs: Vec<StmtRaise>,
}

impl FunctionMetric {
    pub fn new() -> Self {
        Self {
            name: String::new(),
            arguments: Vec::new(),
            if_defs: Vec::new(),
            while_defs: Vec::new(),
            for_defs: Vec::new(),
            return_defs: Vec::new(),
            raise_defs: Vec::new(),
        }
    }

    /// Useful helper in case we want to check if this metric is still empty.
    pub fn is_empty(&self) -> bool {
        self.name.is_empty()
            && self.arguments.is_empty()
            && self.if_defs.is_empty()
            && self.while_defs.is_empty()
            && self.for_defs.is_empty()
            && self.return_defs.is_empty()
            && self.raise_defs.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct ASTAnalyzer {
    pub function_metrics: Vec<FunctionMetric>,
    // Using a stack to support nested (or sequential) function definitions.
    pub function_stack: Vec<FunctionMetric>,
}

impl ASTAnalyzer {
    pub fn new() -> Self {
        Self {
            function_metrics: Vec::new(),
            function_stack: Vec::new(),
        }
    }

    /// Generates the AST from the given Python source code.
    pub fn generate_ast(&self, code: &str) -> Vec<Stmt> {
        Parse::parse_without_path(code).expect("Couldn't parse python code")
    }

    /// A simple entry point to visit each statement.
    pub fn visit(&mut self, nodes: &[Stmt]) {
        for node in nodes {
            self.visit_stmt(node.clone());
        }
    }

    /// Generates python test code based on the collected function metric.
    ///
    /// It generates:
    /// - A test that asserts the function returns the expected literal value if a return statement is present.
    /// - Separate tests that verify a function raises the expected exception if a raise statement is encountered.
    pub fn generate_test(function: &FunctionMetric) -> String {
        let mut test_funcs = Vec::new();
        let func_name = &function.name;
        let args: Vec<String> = function
            .arguments
            .iter()
            .map(|arg| arg.arg.to_string())
            .collect();
        let args_str = args.join(", ");

        // Generate test for normal successful execution if there's a return.
        if !function.return_defs.is_empty() {
            // For simplicity, assume the last return is the intended end-of-execution return.
            if let Some(last_return) = function.return_defs.last() {
                if let Some(literal) = Self::extract_literal_value(last_return.value.as_deref()) {
                    let test_code = format!(
                        r#"def test_{}_success():
    assert {}({}) == {}  # Testing the expected return value
"#,
                        func_name, func_name, args_str, literal
                    );
                    test_funcs.push(test_code);
                }
            }
        }

        // Generate tests for exception paths.
        if !function.raise_defs.is_empty() {
            // Generate a separate test for each raise decision point.
            for (i, raise_stmt) in function.raise_defs.iter().enumerate() {
                if let Some(exc_type) = Self::extract_exception_type(raise_stmt.cause.as_deref()) {
                    let test_code = format!(
                        r#"import pytest

def test_{}_exception_{}():
    with pytest.raises({}):
        {}({})
"#,
                        func_name, i, exc_type, func_name, args_str
                    );
                    test_funcs.push(test_code);
                }
            }
        }

        test_funcs.join("\n")
    }

    /// Attempts to extract a literal value (as a string) from an expression.
    ///
    /// It handles both constant expressions and identifiers.
    fn extract_literal_value(expr_opt: Option<&Expr>) -> Option<String> {
        match expr_opt {
            Some(expr) => match expr {
                Expr::Constant(c) => Some(format!("{:?}", c.value)),
                // If the literal is stored as a variable reference.
                Expr::Name(n) => Some(n.id.to_string()),
                _ => None,
            },
            None => None,
        }
    }

    /// Attempts to extract the exception type from a raise statement’s expression.
    ///
    /// It expects the raise to be of the form:
    /// - `raise ExceptionType(...)` or simply `raise ExceptionType`
    fn extract_exception_type(expr_opt: Option<&Expr>) -> Option<String> {
        match expr_opt {
            Some(expr) => match expr {
                // If the expression is a call (e.g. ExceptionType(...)), extract the function
                // part as the exception type.
                Expr::Call(call_expr) => {
                    if let Expr::Name(ref name_expr) = *call_expr.func {
                        Some(name_expr.id.to_string())
                    } else {
                        None
                    }
                }
                // In case it is just a name (e.g. raise ExceptionType)
                Expr::Name(name_expr) => Some(name_expr.id.to_string()),
                _ => None,
            },
            None => None,
        }
    }
}

impl Visitor for ASTAnalyzer {
    fn visit_stmt_function_def(&mut self, node: StmtFunctionDef) {
        // When entering a new function, create a new metric and push it on the stack.
        let mut func_metric = FunctionMetric::new();
        func_metric.name = node.name.to_string();
        self.function_stack.push(func_metric);

        // Continue visiting the function's body.
        self.generic_visit_stmt_function_def(node);

        // After processing the function, pop the metric from the stack and record it.
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
        if let Some(current) = self.function_stack.last_mut() {
            current.if_defs.push(node.clone());
        }
        self.generic_visit_stmt_if(node);
    }

    fn visit_stmt_while(&mut self, node: StmtWhile) {
        if let Some(current) = self.function_stack.last_mut() {
            current.while_defs.push(node.clone());
        }
        self.generic_visit_stmt_while(node);
    }

    fn visit_stmt_for(&mut self, node: StmtFor) {
        if let Some(current) = self.function_stack.last_mut() {
            current.for_defs.push(node.clone());
        }
        self.generic_visit_stmt_for(node);
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
}
