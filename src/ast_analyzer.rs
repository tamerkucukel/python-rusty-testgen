use rustpython_ast::{Arguments, Stmt, StmtFor, StmtFunctionDef, StmtIf, Visitor};

pub struct ASTAnalyzer {
    functions: Vec<StmtFunctionDef>,
    conditionals: Vec<StmtIf>,
    loops: Vec<StmtFor>,
}

impl<R> Visitor<R> for ASTAnalyzer {
    fn generic_visit_arguments(&mut self, node: Arguments<R>) {}
}

impl ASTAnalyzer {
    pub fn new() -> Self {
        ASTAnalyzer {
            functions: vec![],
            conditionals: vec![],
            loops: vec![],
        }
    }

    pub fn visit(&mut self, nodes: &[Stmt]) {
        nodes.iter().for_each(|node| self.visit_stmt(node.clone()));
    }
}
