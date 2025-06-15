// cfg.rs
// ──────────────────────────────────────────────────────────────────────────────
// Build a *binary-decision* control-flow graph (CFG) for a single Python
// function using rustpython-ast 0.4.  Every `if`, `while` becomes a
// decision node with exactly two ordered successors (TRUE edge = 0,
// FALSE edge = 1).  Every `return` or `raise` becomes an exit node.  The graph
// is small but sufficient for path-based unit-test generation.
// ──────────────────────────────────────────────────────────────────────────────
use rustpython_ast::{
    Arg,
    Arguments,
    Expr,
    ExprName,
    Stmt,
    StmtAssert,
    StmtAssign,
    StmtAugAssign,
    StmtExpr,
    StmtFunctionDef,
    StmtIf,
    // StmtPass is not explicitly used to alter CFG structure beyond collecting it,
    // but keeping it for completeness if it were to be handled differently.
    StmtPass,
    StmtRaise,
    StmtReturn,
    StmtWhile,
    Visitor,
};

use std::collections::HashMap;
// ──────────────────────────────────────────────────────────────────────────────
// ControlFlowGraph – a simple binary-decision control-flow graph
// ──────────────────────────────────────────────────────────────────────────────

/// Represents a unique identifier for a node in the control flow graph.
pub type NodeId = usize;

/// Represents a node in the control flow graph.
#[derive(Clone, Debug)]
pub enum Node {
    /// A conditional node (e.g., `if`, `while` condition).
    Cond {
        /// Statements accumulated in the basic block leading to this condition.
        stmts: Vec<Stmt>,
        /// The conditional expression.
        expr: Expr,
        /// Successor NodeIds: `succ[0]` for True, `succ[1]` for False.
        succ: [NodeId; 2],
    },
    /// A return statement node.
    Return {
        /// Statements accumulated in the basic block leading to this return.
        stmts: Vec<Stmt>,
        /// The `StmtReturn` AST node.
        stmt: StmtReturn,
    },
    /// A raise statement node.
    Raise {
        /// Statements accumulated in the basic block leading to this raise.
        stmts: Vec<Stmt>,
        /// The `StmtRaise` AST node.
        stmt: StmtRaise,
    },
}

/// Represents an edge type in the control flow graph.
#[derive(Clone, Debug, PartialEq, Eq)] // Added PartialEq, Eq for potential comparisons
pub enum Edge {
    /// Edge taken when a condition is true.
    True,
    /// Edge taken when a condition is false.
    False,
    /// Edge representing termination of a path (e.g., from a Return or Raise node).
    Terminal,
}

/// Represents a control flow graph for a Python function.
#[derive(Clone, Debug, Default)]
pub struct ControlFlowGraph {
    /// The entry `NodeId` of the graph.
    entry: NodeId,
    /// The graph structure, mapping `NodeId` to `Node`.
    graph: HashMap<NodeId, Node>,
    /// Stack to keep track of frontier nodes and the edge type leading to them,
    /// used for connecting nodes during CFG construction.
    frontier_stack: Vec<(NodeId, Edge)>,
    /// List of function arguments (name, optional type hint string).
    arguments: Vec<(String, Option<String>)>,
    /// Accumulator for statements within the current basic block being processed.
    current_block_stmts: Vec<Stmt>,
    /// Optional return type hint string of the function.
    fn_return_type: Option<String>, // NEW FIELD
}

impl ControlFlowGraph {
    /// Creates a new, empty `ControlFlowGraph`.
    pub fn new() -> Self {
        // Default::default() is equivalent and idiomatic for structs implementing Default.
        Default::default()
    }

    /// Builds the CFG from a `StmtFunctionDef` AST node.
    /// This consumes the `ControlFlowGraph`'s visitor state.
    pub fn from_ast(&mut self, node: StmtFunctionDef) {
        // The entry node is implicitly the start of the function's body.
        // The first node created by visit_stmt_function_def will become the effective entry.
        self.visit_stmt_function_def(node);
    }

    /// Returns the entry `NodeId` of the control flow graph.
    pub fn get_entry(&self) -> NodeId {
        self.entry
        // Note: self.entry is 0 by default. It's implicitly the first node added.
        // If no nodes are added (e.g. empty function), this might not be meaningful.
        // Consider if entry should be Option<NodeId> or set explicitly after first node.
    }

    /// Returns an immutable reference to the graph data (`HashMap<NodeId, Node>`).
    pub fn get_graph(&self) -> &HashMap<NodeId, Node> {
        &self.graph
    }

    /// Returns an immutable reference to the list of function arguments.
    pub fn get_arguments(&self) -> &Vec<(String, Option<String>)> {
        &self.arguments
    }

    /// Returns an immutable reference to the function's return type hint, if any.
    pub fn get_fn_return_type(&self) -> Option<&String> {
        // NEW METHOD
        self.fn_return_type.as_ref()
    }

    /// Returns an immutable reference to a specific `Node` by its `NodeId`.
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.graph.get(&id)
    }

    /// Returns a mutable reference to the graph data.
    fn get_graph_mut(&mut self) -> &mut HashMap<NodeId, Node> {
        &mut self.graph
    }

    /// Returns a mutable reference to a specific `Node` by its `NodeId`.
    fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.graph.get_mut(&id)
    }

    /// Adds a new node to the control flow graph and returns its ID.
    /// Node IDs are assigned sequentially based on the current graph size.
    fn add_node_internal(&mut self, node: Node) -> NodeId {
        let id = self.graph.len();
        self.graph.insert(id, node);
        if id == 0 {
            // Set the entry point if this is the first node
            self.entry = id;
        }
        id
    }

    /// Connects nodes on the `frontier_stack` to the specified `node_id`.
    /// This is used to link preceding conditional branches to their successor nodes.
    fn connect_frontier(&mut self, node_id: NodeId) {
        // Collect the frontier stack items first to avoid borrowing conflicts
        let frontier_items: Vec<_> = self.frontier_stack.drain(..).rev().collect();

        // Iterate over the collected frontier items and connect nodes.
        for (frontier_node_id, edge) in frontier_items {
            if let Some(node) = self.get_node_mut(frontier_node_id) {
                if let Node::Cond { succ, .. } = node {
                    match edge {
                        Edge::True => succ[0] = node_id,
                        Edge::False => succ[1] = node_id,
                        Edge::Terminal => {
                            // This case should ideally not happen if logic is correct,
                            // as Terminal edges shouldn't be on the frontier for Cond nodes.
                            // Consider logging a warning or an error here if it occurs.
                            // e.g., eprintln!("Warning: Terminal edge found on frontier for Cond node {}", frontier_node_id);
                        }
                    }
                }
                // If it's not a Cond node, it has no successors to connect in this manner.
            }
        }
    }

    /// Extracts argument names and their type hints from the `Arguments` AST node.
    fn extract_function_arguments(&mut self, args: &Arguments) {
        // Helper to process a single argument.
        let mut add_arg = |arg_node: &Arg| {
            let name = arg_node.arg.to_string();
            // Attempt to extract type hint as a simple name (e.g., "int", "str").
            let type_hint = arg_node.annotation.as_ref().and_then(|ann_expr| {
                if let Expr::Name(name_expr) = ann_expr.as_ref() {
                    // Use as_ref() for Box<Expr>
                    Some(name_expr.id.to_string())
                } else {
                    // More complex annotations (e.g., `typing.List[int]`) are not parsed here.
                    None
                }
            });
            self.arguments.push((name, type_hint));
        };

        // Process all argument kinds.
        args.posonlyargs
            .iter()
            .for_each(|arg_with_default| add_arg(&arg_with_default.def));
        args.args
            .iter()
            .for_each(|arg_with_default| add_arg(&arg_with_default.def));
        args.vararg.as_ref().map(|vararg| add_arg(vararg));
        args.kwonlyargs
            .iter()
            .for_each(|arg_with_default| add_arg(&arg_with_default.def));
        args.kwarg.as_ref().map(|kwarg| add_arg(kwarg));
    }

    /// Creates and adds a `Node::Cond` to the graph.
    /// Consumes `self.current_block_stmts`.
    fn add_cond_node(&mut self, expr: Expr, succ: [NodeId; 2]) -> NodeId {
        let node = Node::Cond {
            // Take the accumulated statements for this block.
            stmts: std::mem::take(&mut self.current_block_stmts),
            expr,
            succ,
        };
        self.add_node_internal(node)
    }

    /// Creates and adds a `Node::Return` to the graph.
    /// Consumes `self.current_block_stmts`.
    fn add_return_node(&mut self, stmt: StmtReturn) -> NodeId {
        let node = Node::Return {
            stmts: std::mem::take(&mut self.current_block_stmts),
            stmt,
        };
        self.add_node_internal(node)
    }

    /// Creates and adds a `Node::Raise` to the graph.
    /// Consumes `self.current_block_stmts`.
    fn add_raise_node(&mut self, stmt: StmtRaise) -> NodeId {
        let node = Node::Raise {
            stmts: std::mem::take(&mut self.current_block_stmts),
            stmt,
        };
        self.add_node_internal(node)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Visitor impl – Handles statements that affect control flow or are part of basic blocks.
// ──────────────────────────────────────────────────────────────────────────────

impl Visitor for ControlFlowGraph {
    /// Visits the main function definition.
    fn visit_stmt_function_def(&mut self, node: StmtFunctionDef) {
        self.current_block_stmts.clear(); // Ensure fresh start for function statements.
        self.extract_function_arguments(&node.args);

        // Extract and store the return type hint
        if let Some(return_annotation_expr) = &node.returns {
            if let Expr::Name(ExprName { id, .. }) = return_annotation_expr.as_ref() {
                self.fn_return_type = Some(id.to_string());
            }
            // else: More complex return annotations (e.g., typing.List[int]) are not stored as simple strings.
            // For now, we only capture simple name annotations like "int", "float", "str".
        }

        // The first node created will implicitly be the entry if graph is empty.
        // The `entry` field is set in `add_node_internal`.

        // Process each statement in the function body.
        for stmt in node.body {
            self.visit_stmt(stmt);
        }

        // After processing all explicit statements, handle paths that would implicitly return None.
        // Create a single Node::Return for "return None".
        // Any statements remaining in self.current_block_stmts are those that execute immediately
        // before this implicit return.
        // All open paths on self.frontier_stack (e.g., from if/else branches that didn't explicitly return)
        // will be connected to this single implicit return node.

        // Define the AST for "return None"
        let implicit_return_stmt_ast = StmtReturn {
            value: None,
            range: Default::default(),
        };

        // add_return_node consumes self.current_block_stmts to form the statements list
        // for the new Node::Return, and then adds this new node to the graph.
        let implicit_return_node_id = self.add_return_node(implicit_return_stmt_ast);

        // connect_frontier takes all pending paths from self.frontier_stack
        // and connects them to this newly created implicit_return_node_id.
        // This effectively ensures all paths terminate.
        self.connect_frontier(implicit_return_node_id);

        // Note: If the function explicitly ended with a return/raise, current_block_stmts would be empty
        // (consumed by that explicit return/raise's add_xxx_node call) and frontier_stack would also be empty
        // (drained by the connect_frontier call for that explicit return/raise).
        // In such a case, add_return_node here would create a Node::Return with empty stmts,
        // and connect_frontier would do nothing, which is correct.
    }

    // Accumulate non-control-flow statements.
    fn visit_stmt_expr(&mut self, node: StmtExpr) {
        self.current_block_stmts.push(Stmt::Expr(node.clone()));
        // rustpython_ast::visitor::walk_expr_stmt(self, &node); // Use &node if not consuming
        // If generic_visit_* is intended to call the default Visitor methods to recurse,
        // ensure that's the desired behavior. For simple accumulation, it might not be needed.
        // The original code called self.generic_visit_stmt_expr(node) which consumes node.
        // If node is not meant to be consumed, pass by reference.
        // For just accumulating, no further walk is strictly necessary here.
    }

    fn visit_stmt_assign(&mut self, node: StmtAssign) {
        self.current_block_stmts.push(Stmt::Assign(node.clone()));
        // rustpython_ast::visitor::walk_assign_stmt(self, &node);
    }

    fn visit_stmt_aug_assign(&mut self, node: StmtAugAssign) {
        self.current_block_stmts.push(Stmt::AugAssign(node.clone()));
        // rustpython_ast::visitor::walk_aug_assign_stmt(self, &node);
    }

    fn visit_stmt_assert(&mut self, node: StmtAssert) {
        self.current_block_stmts.push(Stmt::Assert(node.clone()));
        // rustpython_ast::visitor::walk_assert_stmt(self, &node);
    }

    fn visit_stmt_pass(&mut self, node: StmtPass) {
        self.current_block_stmts.push(Stmt::Pass(node.clone()));
        // No sub-expressions to visit in StmtPass.
    }

    // Handle control-flow statements.
    fn visit_stmt_if(&mut self, node: StmtIf) {
        // The conditional node itself. `current_block_stmts` are for the block *before* this if.
        let cond_node_id = self.add_cond_node((*node.test).clone(), [0, 0]); // Placeholder successors
        self.connect_frontier(cond_node_id); // Connect previous paths to this condition.

        // No need to self.visit_expr(*node.test) here, as it's part of the Cond node.
        // The `expr` field in `Node::Cond` holds the test expression.

        // True Branch
        self.frontier_stack.push((cond_node_id, Edge::True));
        for stmt_in_body in node.body {
            self.visit_stmt(stmt_in_body);
        }
        // Collect all paths that fall through the true branch.
        let true_fallthroughs = self.frontier_stack.drain(..).collect::<Vec<_>>();

        // False Branch (orelse)
        self.frontier_stack.push((cond_node_id, Edge::False));
        if !node.orelse.is_empty() {
            for stmt_in_orelse in node.orelse {
                self.visit_stmt(stmt_in_orelse);
            }
        }
        // Collect all paths that fall through the false branch (or the if itself if no orelse).
        let false_fallthroughs = self.frontier_stack.drain(..).collect::<Vec<_>>();

        // Restore frontier: all paths that continued after the if/else.
        self.frontier_stack.extend(true_fallthroughs);
        self.frontier_stack.extend(false_fallthroughs);
    }

    fn visit_stmt_while(&mut self, node: StmtWhile) {
        // Node for the while condition.
        let cond_node_id = self.add_cond_node((*node.test).clone(), [0, 0]);
        self.connect_frontier(cond_node_id);
        // self.visit_expr(*node.test); // Condition is part of Cond node.

        // True branch (loop body)
        self.frontier_stack.push((cond_node_id, Edge::True));
        for stmt_in_body in node.body {
            self.visit_stmt(stmt_in_body);
        }
        // After loop body, paths connect back to the condition.
        self.connect_frontier(cond_node_id);
        // Any break/continue would need more complex handling to modify frontier_stack.

        // False branch (after loop or orelse if present)
        self.frontier_stack.push((cond_node_id, Edge::False));
        if !node.orelse.is_empty() {
            for stmt_in_orelse in node.orelse {
                self.visit_stmt(stmt_in_orelse);
            }
        }
        // The frontier_stack now contains paths exiting the loop.
    }

    fn visit_stmt_return(&mut self, node: StmtReturn) {
        let return_node_id = self.add_return_node(node.clone());
        self.connect_frontier(return_node_id);
        // No need to visit node.value here, it's part of the StmtReturn.
        // The frontier_stack is cleared by connect_frontier for this path.
    }

    fn visit_stmt_raise(&mut self, node: StmtRaise) {
        let raise_node_id = self.add_raise_node(node.clone());
        self.connect_frontier(raise_node_id);
        // No need to visit node.exc or node.cause here.
    }
}
// ──────────────────────────────────────────────────────────────────────────────
