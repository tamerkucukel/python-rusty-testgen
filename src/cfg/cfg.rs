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
    Arg, Arguments, Expr, StmtFunctionDef, StmtIf, StmtRaise, StmtReturn, StmtWhile, Visitor,
}; // Added Arg, Arguments

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
        expr: Expr,
        succ: [NodeId; 2], // Decision edges (index 0 for TRUE, index 1 for FALSE), filled after visiting both branches
    },
    /// Exit node with a return or raise value.
    Return {
        stmt: StmtReturn,
    },

    Raise {
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
    frontier_stack: Vec<(NodeId, Edge)>, // pred edges waiting to connect to next node with scope status.
    arguments: Vec<(String, Option<String>)>, // Stores (arg_name, type_hint_string)
}

// ──────────────────────────────────────────────────────────────────────────────
// Implementation of the ControlFlowGraph
// ──────────────────────────────────────────────────────────────────────────────
impl ControlFlowGraph {
    pub fn new() -> Self {
        Self {
            graph: HashMap::new(),
            entry: 0, // The entry node ID is 0, which is the first node added to the graph.
            frontier_stack: Vec::new(),
            arguments: Vec::new(),
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
}

// ──────────────────────────────────────────────────────────────────────────────
// Visitor impl – only cases that affect control flow
// ──────────────────────────────────────────────────────────────────────────────

impl Visitor for ControlFlowGraph {
    fn visit_stmt_function_def(&mut self, node: StmtFunctionDef) {
        // Extract arguments and their type hints
        self.extract_function_arguments(&node.args);
        self.generic_visit_stmt_function_def(node);
    }

    fn generic_visit_stmt_if(&mut self, node: StmtIf) {
        // Clone the test expression for the Node::Cond, as node.test will be consumed by self.visit_expr.
        let test_expr_for_cond_node = *node.test.clone();

        // Create a new decision node for the `if` statement
        let cond_node = Node::Cond {
            expr: test_expr_for_cond_node,
            succ: [0, 0], // Successors will be determined by connect_frontier calls
                          // from statements within the branches or by fall-through logic.
        };
        // Add the decision node to the graph
        let cond_node_id = self.add_node(cond_node);
        // Connect previous edges from the frontier_stack to this new `if` node.
        // After this, self.frontier_stack should be empty.
        self.connect_frontier(cond_node_id);

        // Visit the condition expression (original code had this, possibly for other analysis).
        // This consumes node.test.
        self.visit_expr(*node.test);

        // --- True Branch ---
        // The frontier for the true branch's body is solely the True edge from cond_node_id.
        // self.frontier_stack is expected to be empty after connect_frontier(cond_node_id).
        self.frontier_stack.push((cond_node_id, Edge::True));
        for stmt in node.body {
            // node.body is Vec<Stmt>, stmt is Stmt (moved)
            self.visit_stmt(stmt);
        }
        // After visiting the body, self.frontier_stack contains all fall-through paths from the true branch.
        let true_fallthroughs = self.frontier_stack.clone();
        self.frontier_stack.clear(); // Clear before processing the false branch

        // --- False Branch (orelse) ---
        // The frontier for the orelse branch's body is solely the False edge from cond_node_id.
        self.frontier_stack.push((cond_node_id, Edge::False));
        if node.orelse.is_empty() {
            // No explicit orelse block. The (cond_node_id, Edge::False) edge itself is a fall-through path.
            // It's currently in self.frontier_stack and will be captured by false_fallthroughs.
        } else {
            for stmt in node.orelse {
                // node.orelse is Vec<Stmt>, stmt is Stmt (moved)
                self.visit_stmt(stmt);
            }
        }
        // After visiting orelse (or if orelse was empty and (cond_node_id, Edge::False) remained),
        // self.frontier_stack contains all fall-through paths from the false branch.
        let false_fallthroughs = self.frontier_stack.clone();
        self.frontier_stack.clear(); // Clear before combining fall-throughs

        // --- Combine fall-throughs ---
        // The new frontier_stack for statements *after* this if-elif-else block
        // consists of fall-throughs from the true branch and fall-throughs from the false branch.
        self.frontier_stack.extend(true_fallthroughs);
        self.frontier_stack.extend(false_fallthroughs);
    }

    fn generic_visit_stmt_while(&mut self, node: StmtWhile) {
        // Create a new decision node for the `while` statement
        let cond_node = Node::Cond {
            expr: *node.test.clone(),
            succ: [0, 0],
        };
        // Add the decision node to the graph
        let cond_node_id = self.add_node(cond_node);
        // Connect previous edges to the node in the frontier.
        self.connect_frontier(cond_node_id);

        // Visit the condition expression
        {
            let value = node.test;
            self.visit_expr(*value);
        }

        // Add true edge of the current node to the frontier stack
        self.frontier_stack.push((cond_node_id, Edge::True));

        for value in node.body {
            self.visit_stmt(value);
        }
        // Add false edge of the current node to the frontier stack
        self.frontier_stack.push((cond_node_id, Edge::False));

        for value in node.orelse {
            self.visit_stmt(value);
        }
    }

    fn visit_stmt_return(&mut self, node: StmtReturn) {
        // Create an exit node for the `return` statement
        let return_node = Node::Return { stmt: node.clone() };
        // Add the exit node to the graph
        let return_node_id = self.add_node(return_node);
        // Connect previous edges to the node in the frontier.
        self.connect_frontier(return_node_id);
    }

    fn visit_stmt_raise(&mut self, node: StmtRaise) {
        // Create an exit node for the `raise` statement
        let raise_node = Node::Raise { stmt: node.clone() };
        // Add the exit node to the graph
        let raise_node_id = self.add_node(raise_node);
        // Connect previous edges to the node in the frontier.
        self.connect_frontier(raise_node_id);
    }
}
// ──────────────────────────────────────────────────────────────────────────────
