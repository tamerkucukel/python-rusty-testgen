// cfg.rs
// ──────────────────────────────────────────────────────────────────────────────
// Build a *binary-decision* control-flow graph (CFG) for a single Python
// function using rustpython-ast 0.4.  Every `if`, `while` becomes a
// decision node with exactly two ordered successors (TRUE edge = 0,
// FALSE edge = 1).  Every `return` or `raise` becomes an exit node.  The graph
// is small but sufficient for path-based unit-test generation.
//
// Compile with: cargo add rustpython-ast = "0.4"
//
// ──────────────────────────────────────────────────────────────────────────────
use rustpython_ast::{
    Arg,
    Arguments,
    Expr,
    Stmt, // Added Stmt
    StmtAssert,
    StmtAssign,
    StmtAugAssign,
    StmtExpr,
    StmtFunctionDef,
    StmtIf,
    StmtPass, // Added StmtAssert, StmtPass
    StmtRaise,
    StmtReturn,
    StmtWhile,
    Visitor,
};

use std::collections::HashMap;
// ──────────────────────────────────────────────────────────────────────────────
// ControlFlowGraph – a simple binary-decision control-flow graph
// Implemented as a map of nodes, each with a unique ID.
// Each node is either a decision node (if, while, for) or an exit node
// (return, raise).
// The graph is built in a single pass using a visitor pattern.
// The graph is not a full CFG, but a simplified version with only binary
// decision nodes and exit nodes.
// ──────────────────────────────────────────────────────────────────────────────

/// Represents a unique identifier for a node in the control flow graph.
pub type NodeId = usize;

/// Represents a node in the control flow graph.
#[derive(Clone, Debug)]
pub enum Node {
    Cond {
        stmts: Vec<Stmt>, // Statements in the block leading to this condition
        expr: Expr,
        succ: [NodeId; 2],
    },
    Return {
        stmts: Vec<Stmt>, // Statements in the block leading to this return
        stmt: StmtReturn,
    },
    Raise {
        stmts: Vec<Stmt>, // Statements in the block leading to this raise
        stmt: StmtRaise,
    },
}

/// Represents a binary decision edge in the control flow graph.
#[derive(Clone, Debug)]
pub enum Edge {
    True,     // TRUE edge (0)
    False,    // FALSE edge (1)
    Terminal, // Terminal edge for exit nodes (return, raise)
}

/// Represents a control flow graph for a Python function.
#[derive(Clone, Debug, Default)]
pub struct ControlFlowGraph {
    entry: NodeId,
    graph: HashMap<NodeId, Node>,
    frontier_stack: Vec<(NodeId, Edge)>,
    arguments: Vec<(String, Option<String>)>,
    current_block_stmts: Vec<Stmt>, // Accumulator for current basic block
}

impl ControlFlowGraph {
    pub fn new() -> Self {
        Self {
            graph: HashMap::new(),
            entry: 0, // The entry node ID is 0, which is the first node added to the graph.
            frontier_stack: Vec::new(),
            arguments: Vec::new(),
            current_block_stmts: Vec::new(),
        }
    }

    /// Visits a function definition and builds the CFG for it.
    pub fn from_ast(&mut self, node: StmtFunctionDef) {
        self.visit_stmt_function_def(node);
    }

    /// Returns the entry node ID of the control flow graph.
    pub fn get_entry(&self) -> NodeId {
        self.entry
    }

    /// Returns a reference to the control flow graph.
    pub fn get_graph(&self) -> &HashMap<NodeId, Node> {
        &self.graph
    }

    /// Returns a reference to the function's arguments (name, type_hint_string).
    pub fn get_arguments(&self) -> &Vec<(String, Option<String>)> {
        &self.arguments
    }

    /// Returns the entry node ID of the control flow graph.
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.graph.get(&id)
    }

    /// Returns a mutable reference to the control flow graph.
    pub fn get_graph_mut(&mut self) -> &mut HashMap<NodeId, Node> {
        &mut self.graph
    }

    /// Returns a mutable reference to a node in the graph by its ID.
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.graph.get_mut(&id)
    }

    /// Adds a new node to the control flow graph and returns its ID.
    fn add_node(&mut self, node: Node) -> NodeId {
        let id = self.graph.len();
        self.graph.insert(id, node);
        id
    }

    /// Connects the current frontier stack to the specified node.
    fn connect_frontier(&mut self, node_id: NodeId) {
        while !self.frontier_stack.is_empty() {
            if let Some((frontier_node_id, edge)) = self.frontier_stack.pop() {
                if let Some(node) = self.get_node_mut(frontier_node_id) {
                    match node {
                        Node::Cond { succ, .. } => {
                            match edge {
                                Edge::True => succ[0] = node_id,  // Connect TRUE edge
                                Edge::False => succ[1] = node_id, // Connect FALSE edge
                                Edge::Terminal => continue, // Terminal edges represent exit points (e.g., return, raise)
                                                            // and are not connected to decision nodes because they
                                                            // signify the end of a control flow path.
                            }
                        }
                        _ => continue, // Only decision nodes have successors
                    }
                }
            }
        }
    }

    /// Helper to extract argument names and type hints.
    fn extract_function_arguments(&mut self, args: &Arguments) {
        // Process positional-only arguments
        for arg_with_default in &args.posonlyargs {
            self.add_argument(&arg_with_default.def);
        }
        // Process regular arguments
        for arg_with_default in &args.args {
            self.add_argument(&arg_with_default.def);
        }
        // Process vararg
        if let Some(vararg) = &args.vararg {
            self.add_argument(vararg);
        }
        // Process keyword-only arguments
        for arg_with_default in &args.kwonlyargs {
            self.add_argument(&arg_with_default.def);
        }
        // Process kwarg
        if let Some(kwarg) = &args.kwarg {
            self.add_argument(kwarg);
        }
    }

    fn add_argument(&mut self, arg: &Arg) {
        let name = arg.arg.to_string();
        let type_hint = arg.annotation.as_ref().and_then(|ann_expr| {
            if let Expr::Name(name_expr) = &**ann_expr {
                Some(name_expr.id.to_string())
            } else {
                // For now, only simple Name annotations like "int", "bool" are parsed.
                // More complex annotations (e.g., `typing.List[int]`) would require deeper parsing.
                None
            }
        });
        self.arguments.push((name, type_hint));
    }

    // Modified add_node helpers to consume current_block_stmts
    fn add_cond_node(&mut self, expr: Expr, succ: [NodeId; 2]) -> NodeId {
        let id = self.graph.len(); // Determine ID before draining
        let node = Node::Cond {
            stmts: self.current_block_stmts.drain(..).collect(),
            expr,
            succ,
        };
        self.graph.insert(id, node);
        id
    }

    fn add_return_node(&mut self, stmt: StmtReturn) -> NodeId {
        let id = self.graph.len();
        let node = Node::Return {
            stmts: self.current_block_stmts.drain(..).collect(),
            stmt,
        };
        self.graph.insert(id, node);
        id
    }

    fn add_raise_node(&mut self, stmt: StmtRaise) -> NodeId {
        let id = self.graph.len();
        let node = Node::Raise {
            stmts: self.current_block_stmts.drain(..).collect(),
            stmt,
        };
        self.graph.insert(id, node);
        id
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Visitor impl – only cases that affect control flow
// ──────────────────────────────────────────────────────────────────────────────

impl Visitor for ControlFlowGraph {
    fn visit_stmt_function_def(&mut self, node: StmtFunctionDef) {
        self.current_block_stmts.clear(); // Ensure fresh start for function statements
        self.extract_function_arguments(&node.args);
        for stmt in node.body {
            self.visit_stmt(stmt); // Process each statement in the function body
        }
        // After processing all statements, if current_block_stmts is not empty,
        // it implies an implicit return. For simplicity, we currently rely on explicit
        // return/raise to terminate paths and clear current_block_stmts.
        // A more robust solution might add an implicit Node::Return here if needed.
    }

    // Override specific statement visitors to accumulate them or handle control flow

    fn visit_stmt_expr(&mut self, node: StmtExpr) {
        self.current_block_stmts.push(Stmt::Expr(node.clone()));
        self.generic_visit_stmt_expr(node);
    }

    fn visit_stmt_assign(&mut self, node: StmtAssign) {
        self.current_block_stmts.push(Stmt::Assign(node.clone()));
        self.generic_visit_stmt_assign(node);
    }

    fn visit_stmt_aug_assign(&mut self, node: StmtAugAssign) {
        self.current_block_stmts.push(Stmt::AugAssign(node.clone()));
        self.generic_visit_stmt_aug_assign(node);
    }

    fn visit_stmt_assert(&mut self, node: StmtAssert) {
        self.current_block_stmts.push(Stmt::Assert(node.clone()));
        self.generic_visit_stmt_assert(node);
    }

    // Control Flow Statement Visitors
    // These methods will use add_cond_node, add_return_node, or add_raise_node,
    // which internally drain self.current_block_stmts.

    fn visit_stmt_if(&mut self, node: StmtIf) {
        let test_expr_for_cond_node = *node.test.clone();
        let cond_node_id = self.add_cond_node(test_expr_for_cond_node, [0, 0]); // Drains stmts before if
        self.connect_frontier(cond_node_id);

        self.visit_expr(*node.test); // Visit the condition expression itself

        // True Branch
        self.frontier_stack.push((cond_node_id, Edge::True));
        for stmt_in_body in node.body {
            self.visit_stmt(stmt_in_body);
        }
        let true_fallthroughs = self.frontier_stack.clone();
        self.frontier_stack.clear();

        // False Branch (orelse)
        self.frontier_stack.push((cond_node_id, Edge::False));
        if !node.orelse.is_empty() {
            for stmt_in_orelse in node.orelse {
                self.visit_stmt(stmt_in_orelse);
            }
        }
        let false_fallthroughs = self.frontier_stack.clone();
        self.frontier_stack.clear();

        // Combine fall-throughs
        self.frontier_stack.extend(true_fallthroughs);
        self.frontier_stack.extend(false_fallthroughs);
    }

    fn visit_stmt_while(&mut self, node: StmtWhile) {
        // Simplified handling for `while` for now.
        // The block before the `while` condition.
        let test_expr_for_cond_node = *node.test.clone();
        let cond_node_id = self.add_cond_node(test_expr_for_cond_node, [0, 0]);
        self.connect_frontier(cond_node_id);
        self.visit_expr(*node.test);

        // True branch (loop body)
        self.frontier_stack.push((cond_node_id, Edge::True));
        for stmt_in_body in node.body {
            self.visit_stmt(stmt_in_body);
        }
        // After loop body, paths connect back to the condition.
        self.connect_frontier(cond_node_id);

        // False branch (after loop or orelse)
        // The frontier for "after loop" is the False edge from cond_node_id.
        self.frontier_stack.push((cond_node_id, Edge::False));
        if !node.orelse.is_empty() {
            for stmt_in_orelse in node.orelse {
                self.visit_stmt(stmt_in_orelse);
            }
        }
        // The frontier_stack now contains paths exiting the loop (either from orelse or directly if orelse is empty).
    }

    fn visit_stmt_return(&mut self, node: StmtReturn) {
        let return_node_id = self.add_return_node(node.clone()); // Drains stmts before return
        self.connect_frontier(return_node_id);
        if let Some(val) = &node.value {
            self.visit_expr((**val).clone());
        }
    }

    fn visit_stmt_raise(&mut self, node: StmtRaise) {
        let raise_node_id = self.add_raise_node(node.clone()); // Drains stmts before raise
        self.connect_frontier(raise_node_id);
        if let Some(exc) = &node.exc {
            self.visit_expr((**exc).clone());
        }
        if let Some(cause) = &node.cause {
            self.visit_expr((**cause).clone());
        }
    }
}
// ──────────────────────────────────────────────────────────────────────────────
