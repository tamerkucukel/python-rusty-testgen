use std::collections::HashMap;
use std::fmt::Write as FmtWrite; // Alias to avoid conflict with std::io::Write
use std::io::Write;

use crate::cfg::{ControlFlowGraph, Edge, Node, NodeId};
use rustpython_ast::{
    Arg, BoolOp, CmpOp, Constant, Expr, ExprBinOp, ExprBoolOp, ExprCall, ExprCompare,
    ExprConstant, ExprName, ExprUnaryOp, Keyword, Operator, Stmt, StmtAssert, StmtAssign,
    StmtAugAssign, StmtExpr, StmtPass, StmtRaise, StmtReturn, UnaryOp,
};

// Helper function to format an Expr AST node into a Python-like string
fn format_expr(expr: &Expr) -> String {
    match &expr {
        Expr::Constant(ExprConstant { value, .. }) => format_constant(value),
        Expr::Name(ExprName { id, .. }) => id.to_string(),
        Expr::BinOp(ExprBinOp { left, op, right, .. }) => {
            format!(
                "({}) {} ({})",
                format_expr(left),
                format_operator(op),
                format_expr(right)
            )
        }
        Expr::UnaryOp(ExprUnaryOp { op, operand, .. }) => {
            format!("{}{}", format_unary_operator(op), format_expr(operand)) // No space for unary like -x or not x
        }
        Expr::BoolOp(ExprBoolOp { op, values, .. }) => values
            .iter()
            .map(format_expr)
            .collect::<Vec<String>>()
            .join(&format!(" {} ", format_bool_operator(op))),
        Expr::Compare(ExprCompare {
            left,
            ops,
            comparators,
            ..
        }) => {
            let mut s = format_expr(left);
            for (op, comp) in ops.iter().zip(comparators.iter()) {
                s.push_str(&format!(" {} {}", format_cmp_operator(op), format_expr(comp)));
            }
            s
        }
        Expr::Call(ExprCall {
            func,
            args,
            keywords,
            ..
        }) => {
            let func_str = format_expr(func);
            let args_str = args.iter().map(format_expr).collect::<Vec<String>>().join(", ");
            let keywords_str = keywords
                .iter()
                .map(|kw| {
                    format!(
                        "{}={}",
                        kw.arg.as_ref().map_or("?".to_string(), |s| s.to_string()),
                        format_expr(&kw.value)
                    )
                })
                .collect::<Vec<String>>()
                .join(", ");
            if keywords_str.is_empty() {
                format!("{}({})", func_str, args_str)
            } else if args_str.is_empty() {
                format!("{}({})", func_str, keywords_str)
            } else {
                format!("{}({}, {})", func_str, args_str, keywords_str)
            }
        }
        // Add more common Expr types as needed
        _ => format!("<expr: {:?}>", expr), // Fallback to debug representation of the kind
    }
}

fn format_constant(constant: &Constant) -> String {
    match constant {
        Constant::Int(i) => i.to_string(),
        Constant::Float(f) => f.to_string(),
        Constant::Bool(b) => if *b { "True".to_string() } else { "False".to_string() },
        Constant::Str(s) => format!("\"{}\"", s.to_string().escape_default()), // Use .to_string() for LocatedString
        Constant::None => "None".to_string(),
        Constant::Ellipsis => "...".to_string(),
        Constant::Bytes(b) => format!("b\"{}\"", String::from_utf8_lossy(b).escape_default()),
        _ => format!("<const: {:?}>", constant),
    }
}

fn format_operator(op: &Operator) -> String {
    match op {
        Operator::Add => "+".to_string(),
        Operator::Sub => "-".to_string(),
        Operator::Mult => "*".to_string(),
        Operator::MatMult => "@".to_string(),
        Operator::Div => "/".to_string(),
        Operator::Mod => "%".to_string(),
        Operator::Pow => "**".to_string(),
        Operator::LShift => "<<".to_string(),
        Operator::RShift => ">>".to_string(),
        Operator::BitOr => "|".to_string(),
        Operator::BitXor => "^".to_string(),
        Operator::BitAnd => "&".to_string(),
        Operator::FloorDiv => "//".to_string(),
    }
}

fn format_unary_operator(op: &UnaryOp) -> String {
    match op {
        UnaryOp::Invert => "~".to_string(),
        UnaryOp::Not => "not ".to_string(), // Add space for "not x"
        UnaryOp::UAdd => "+".to_string(),
        UnaryOp::USub => "-".to_string(),
    }
}

fn format_bool_operator(op: &BoolOp) -> String {
    match op {
        BoolOp::And => "and".to_string(),
        BoolOp::Or => "or".to_string(),
    }
}

fn format_cmp_operator(op: &CmpOp) -> String {
    match op {
        CmpOp::Eq => "==".to_string(),
        CmpOp::NotEq => "!=".to_string(),
        CmpOp::Lt => "<".to_string(),
        CmpOp::LtE => "<=".to_string(),
        CmpOp::Gt => ">".to_string(),
        CmpOp::GtE => ">=".to_string(),
        CmpOp::Is => "is".to_string(),
        CmpOp::IsNot => "is not".to_string(),
        CmpOp::In => "in".to_string(),
        CmpOp::NotIn => "not in".to_string(),
    }
}

// Helper function to format a Stmt AST node into a Python-like string
fn format_stmt(stmt: &Stmt) -> String {
    match &stmt {
        Stmt::Assign(StmtAssign {
            targets,
            value,
            type_comment: _, // Ignoring type_comment for simplicity
            ..
        }) => {
            let targets_str = targets.iter().map(format_expr).collect::<Vec<String>>().join(", ");
            format!("{} = {}", targets_str, format_expr(value))
        }
        Stmt::Expr(StmtExpr { value, .. }) => format_expr(value),
        Stmt::Assert(StmtAssert { test, msg, .. }) => {
            if let Some(m) = msg {
                format!("assert {}, {}", format_expr(test), format_expr(m))
            } else {
                format!("assert {}", format_expr(test))
            }
        }
        Stmt::Pass(_) => "pass".to_string(),
        Stmt::AugAssign(StmtAugAssign {
            target,
            op,
            value,
            ..
        }) => {
            format!(
                "{} {}= {}",
                format_expr(target),
                format_operator(op), // AugAssign uses Operator
                format_expr(value)
            )
        }
        // Add more common Stmt types as needed
        _ => format!("<stmt: {:?}>", stmt), // Fallback to debug representation of the kind
    }
}

/// `PathScraper` is responsible for finding all execution paths in a `ControlFlowGraph`.
pub struct PathScraper;

impl PathScraper {
    /// Returns all paths in the control flow graph.
    ///
    /// A path is represented as a vector of `(NodeId, Edge)` tuples,
    /// indicating the sequence of nodes visited and the edges taken.
    /// Returns `None` if the graph is empty or no paths are found.
    pub fn get_paths(control_flow_graph: &ControlFlowGraph) -> Option<Vec<Vec<(NodeId, Edge)>>> {
        // Call the traversal logic.
        let paths = Self::traverse(
            control_flow_graph.get_graph(),
            control_flow_graph.get_entry(),
        );

        // Return None if no paths were found, otherwise Some(paths).
        // Simplified conditional return.
        if paths.is_empty() {
            None
        } else {
            Some(paths)
        }
    }

    /// Prints all paths and nodes in the control flow graph to the given writer.
    pub fn print_paths_to_writer(
        control_flow_graph: &ControlFlowGraph,
        writer: &mut dyn Write,
    ) -> std::io::Result<()> {
        if let Some(paths) = Self::get_paths(control_flow_graph) {
            writeln!(writer, "=== CONTROL FLOW GRAPH PATHS ===")?;
            writeln!(writer, "Total paths found: {}", paths.len())?;
            writeln!(writer)?;

            for (i, path) in paths.iter().enumerate() {
                writeln!(writer, "Path {}: {:?}", i, path)?;
            }
            writeln!(writer)?;
        } else {
            writeln!(writer, "No paths found in the control flow graph.")?;
            writeln!(writer)?;
        }

        writeln!(writer, "=== CONTROL FLOW GRAPH NODES ===")?;
        let mut node_ids: Vec<_> = control_flow_graph.get_graph().keys().collect();
        node_ids.sort(); // Print nodes in a consistent order

        for node_id_ref in node_ids {
            let node_id = *node_id_ref;
            if let Some(node) = control_flow_graph.get_node(node_id) {
                writeln!(writer, "Node ID: {}", node_id)?;
                match node {
                    Node::Cond { stmts, expr, succ } => {
                        writeln!(writer, "  Type: Conditional")?;
                        for s in stmts {
                            writeln!(writer, "    Stmt: {}", format_stmt(s))?;
                        }
                        writeln!(writer, "    Condition: {}", format_expr(expr))?;
                        writeln!(writer, "    Successors: True -> {}, False -> {}", succ[0], succ[1])?;
                    }
                    Node::Return { stmts, stmt } => {
                        writeln!(writer, "  Type: Return")?;
                        for s in stmts {
                            writeln!(writer, "    Stmt: {}", format_stmt(s))?;
                        }
                        if let Some(value_expr) = &stmt.value {
                            writeln!(writer, "    Value: {}", format_expr(value_expr))?;
                        } else {
                            writeln!(writer, "    Value: None")?;
                        }
                    }
                    Node::Raise { stmts, stmt } => {
                        writeln!(writer, "  Type: Raise")?;
                        for s in stmts {
                            writeln!(writer, "    Stmt: {}", format_stmt(s))?;
                        }
                        if let Some(exc_expr) = &stmt.exc {
                            writeln!(writer, "    Exception: {}", format_expr(exc_expr))?;
                        }
                        if let Some(cause_expr) = &stmt.cause {
                            writeln!(writer, "    Cause: {}", format_expr(cause_expr))?;
                        }
                    }
                }
                writeln!(writer)?; // Add a blank line after each node's details
            }
        }
        Ok(())
    }

    /// Performs a depth-first traversal to collect all root-to-leaf paths.
    ///
    /// `graph`: The graph structure to traverse.
    /// `entry`: The starting `NodeId` for the traversal.
    /// Returns a vector of all found paths.
    fn traverse(graph: &HashMap<NodeId, Node>, entry: NodeId) -> Vec<Vec<(NodeId, Edge)>> {
        let mut all_paths: Vec<Vec<(NodeId, Edge)>> = Vec::new();
        // Stack stores tuples of (current_node_id, path_taken_to_reach_current_node).
        let mut stack: Vec<(NodeId, Vec<(NodeId, Edge)>)> = Vec::new();

        // Start traversal from the entry node with an empty path.
        stack.push((entry, Vec::new()));

        while let Some((current_node_id, mut path_so_far)) = stack.pop() {
            // path_so_far is now mutable
            // Get the current node from the graph.
            if let Some(node) = graph.get(&current_node_id) {
                match node {
                    Node::Cond {
                        expr: _,
                        stmts: _,
                        succ,
                    } => {
                        // stmts and expr are not used in this traversal logic
                        // Explore the false branch.
                        // Cloned path_so_far for the false branch, current path_so_far is used for true branch.
                        let mut false_path = path_so_far.clone();
                        false_path.push((current_node_id, Edge::False));
                        stack.push((succ[1], false_path));

                        // Explore the true branch.
                        path_so_far.push((current_node_id, Edge::True));
                        stack.push((succ[0], path_so_far));
                    }
                    Node::Return { .. } | Node::Raise { .. } => {
                        // Current node is a terminal node (Return or Raise).
                        // Add the terminal edge and store the completed path.
                        path_so_far.push((current_node_id, Edge::Terminal));
                        all_paths.push(path_so_far);
                    }
                }
            }
            // If a node_id is not in the graph, that path silently terminates.
            // This could happen if the CFG is malformed or incomplete.
        }
        all_paths
    }
}
