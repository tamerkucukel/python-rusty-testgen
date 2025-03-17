use rustpython_ast::{
    Arg, Arguments, Expr, Stmt, StmtFor, StmtFunctionDef, StmtIf, StmtReturn, StmtWhile, Visitor,
};
use rustpython_parser::Parse;

// Holds each function decision points into single structure.
#[derive(Clone, Debug)]
pub struct FunctionMetric {
    pub name: String,
    pub arguments: Vec<Arg>,
    pub if_defs: Vec<StmtIf>,
    pub while_defs: Vec<StmtWhile>,
    pub for_defs: Vec<StmtFor>,
    pub return_defs: Vec<StmtReturn>,
}

impl FunctionMetric {
    pub fn new() -> Self {
        Self {
            name: String::new(),
            arguments: vec![],
            if_defs: vec![],
            while_defs: vec![],
            for_defs: vec![],
            return_defs: vec![],
        }
    }
}

#[derive(Clone, Debug)]
pub struct ASTAnalyzer {
    pub function_metrics: Vec<FunctionMetric>,
    pub current_function: FunctionMetric,
}

impl Visitor for ASTAnalyzer {
    fn visit_stmt_function_def(&mut self, node: StmtFunctionDef) {
        self.current_function.name = node.name.to_string();
        self.generic_visit_stmt_function_def(node)
    }

    fn visit_arguments(&mut self, node: Arguments) {
        for arg in &node.args {
            self.current_function.arguments.push(arg.def.clone());
        }
        self.generic_visit_arguments(node)
    }
    fn visit_stmt_if(&mut self, node: StmtIf) {
        self.current_function.if_defs.push(node.to_owned());
        self.generic_visit_stmt_if(node)
    }

    fn visit_stmt_while(&mut self, node: StmtWhile) {
        self.current_function.while_defs.push(node.to_owned());
        self.generic_visit_stmt_while(node)
    }

    fn visit_stmt_for(&mut self, node: StmtFor) {
        self.current_function.for_defs.push(node.to_owned());
        self.generic_visit_stmt_for(node)
    }

    fn visit_stmt_return(&mut self, node: StmtReturn) {
        self.current_function.return_defs.push(node.to_owned());
        self.generic_visit_stmt_return(node)
    }
}

impl ASTAnalyzer {
    pub fn new() -> Self {
        Self {
            function_metrics: vec![],
            current_function: FunctionMetric::new(),
        }
    }

    pub fn generate_ast(&self, code: &str) -> Vec<Stmt> {
        Parse::parse_without_path(code).expect("Couldn't parse python code")
    }

    pub fn visit(&mut self, nodes: &[Stmt]) {
        nodes.iter().for_each(|node| self.visit_stmt(node.clone()));
        self.function_metrics.push(self.current_function.to_owned());
        self.current_function = FunctionMetric::new()
    }

    pub fn generate_test(function: &FunctionMetric) -> String {
        let mut test_cases = vec![];
        let func_name = function.name.clone();
        let args: Vec<String> = function
            .arguments
            .iter()
            .map(|arg| arg.arg.to_string())
            .collect();
        let args_str = args.join(", ");

        // If conditions exist, generate tests based on them
        if !function.if_defs.is_empty() {
            for if_stmt in &function.if_defs {
                println!("Compare checked !");
                match &*if_stmt.test {
                    Expr::Compare(expr_compare) => {
                        println!("Compare matched !");
                        if let (Some(left_var), Some(right_var)) = (
                            Self::extract_identifier(&expr_compare.left),
                            Self::extract_identifier(&expr_compare.comparators[0]),
                        ) {
                            // Generate expected values based on conditions
                            let expected_eq = left_var.clone(); // If a == b, return a
                            let expected_gt = left_var.clone(); // If a > b, return a
                            let expected_lt = right_var.clone(); // If a < b, return b

                            test_cases.push(format!(
                                "assert {}({}, {}) == {}  # Test equality",
                                func_name, left_var, right_var, expected_eq
                            ));
                            test_cases.push(format!(
                                "assert {}({}, {}) == {}  # Test greater than",
                                func_name, left_var, right_var, expected_gt
                            ));
                            test_cases.push(format!(
                                "assert {}({}, {}) == {}  # Test less than",
                                func_name, right_var, left_var, expected_lt
                            ));
                        }
                    }
                    _ => {
                        panic!("Not implemented.")
                    }
                }
            }
        }

        // Default return case
        if !function.return_defs.is_empty() {
            for ret_stmt in &function.return_defs {
                if let Some(return_value) = &ret_stmt.value {
                    if let Some(return_var) = Self::extract_identifier(return_value) {
                        test_cases.push(format!(
                            "assert {}({}) == {}  # Test return value",
                            func_name, args_str, return_var
                        ));
                    }
                }
            }
        }

        // Combine into a Python test function
        let test_code = format!(
            r#"def test_{}():
    {}
            "#,
            func_name,
            test_cases.join("\n    ")
        );

        test_code
    }

    fn extract_identifier(expr: &rustpython_ast::Expr) -> Option<String> {
        return Some(expr.as_name_expr().unwrap().id.to_string());
    }
}
