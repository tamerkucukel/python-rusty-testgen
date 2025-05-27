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
use rustpython_ast::{Expr, StmtFunctionDef, StmtIf, StmtRaise, StmtReturn, StmtWhile, Visitor};

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

    /// Prints all paths in the control flow graph.
    pub fn print_paths(&self) {
        let mut current_path = Vec::new();

        fn print_path(
            graph: &ControlFlowGraph,
            node_id: NodeId,
            current_path: &mut Vec<(NodeId, Edge)>,
        ) {
            if let Some(node) = graph.get_node(node_id) {
                match node {
                    Node::Cond { succ, .. } => {
                        // Traverse TRUE edge first
                        current_path.push((node_id, Edge::True));
                        print_path(graph, succ[0], current_path);
                        current_path.pop(); // Pop TRUE edge

                        // Then FALSE edge
                        current_path.push((node_id, Edge::False));
                        print_path(graph, succ[1], current_path);
                        current_path.pop(); // Pop FALSE edge
                    }
                    Node::Return { .. } | Node::Raise { .. } => {
                        // Print the path when reaching an exit node
                        current_path.push((node_id, Edge::True)); // Mark the exit node
                        println!("Path: {:?}", current_path);
                        current_path.pop(); // Pop the exit node marking
                    }
                }
            }
        }
        print_path(self, self.entry, &mut current_path);
        println!("Nodes in the graph: {:#?}", self.graph);
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
}

// ──────────────────────────────────────────────────────────────────────────────
// Visitor impl – only cases that affect control flow
// ──────────────────────────────────────────────────────────────────────────────

impl Visitor for ControlFlowGraph {
    fn generic_visit_stmt_if(&mut self, node: StmtIf) {
        // Create a new decision node for the `if` statement
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
