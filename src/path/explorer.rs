use std::collections::HashMap;
// use std::fmt::Write as FmtWrite; // Not strictly needed if only using std::io::Write
use std::io::Write;

use crate::cfg::{ControlFlowGraph, Edge, Node, NodeId};
use rustpython_ast::{
    // Imports based on the attachment's usage in format_expr/format_stmt
    BoolOp,
    CmpOp,
    Constant,
    Expr,
    ExprBinOp,
    ExprBoolOp,
    ExprCall,
    ExprCompare,
    ExprConstant,
    ExprName,
    ExprUnaryOp,
    Operator,
    Stmt,
    StmtAssert,
    StmtAssign,
    StmtAugAssign,
    StmtExpr,
    UnaryOp,
};

// format_expr as per attachment structure, not using .node for matching
fn format_expr(expr: &Expr) -> String {
    // Assuming Expr is the enum itself, not Located<Expr_> for this formatter's logic
    // This matches the style of the provided attachment.
    match expr {
        Expr::Constant(ExprConstant { value, .. }) => format_constant(value),
        Expr::Name(ExprName { id, .. }) => id.to_string(),
        Expr::BinOp(ExprBinOp {
            left, op, right, ..
        }) => {
            format!(
                "({}) {} ({})",
                format_expr(left),
                format_operator(op),
                format_expr(right)
            )
        }
        Expr::UnaryOp(ExprUnaryOp { op, operand, .. }) => {
            format!("{}{}", format_unary_operator(op), format_expr(operand))
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
                s.push_str(&format!(
                    " {} {}",
                    format_cmp_operator(op),
                    format_expr(comp)
                ));
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
            let args_str_vec: Vec<String> = args.iter().map(format_expr).collect();
            let keywords_str_vec: Vec<String> = keywords
                .iter()
                .map(|kw| {
                    format!(
                        "{}={}",
                        kw.arg
                            .as_ref()
                            .map_or_else(|| "?".to_string(), |id| id.to_string()),
                        format_expr(&kw.value)
                    )
                })
                .collect();

            let mut all_args_parts = Vec::new();
            if !args_str_vec.is_empty() {
                all_args_parts.push(args_str_vec.join(", "));
            }
            if !keywords_str_vec.is_empty() {
                all_args_parts.push(keywords_str_vec.join(", "));
            }
            format!("{}({})", func_str, all_args_parts.join(", "))
        }
        // Other Expr variants from rustpython_ast::Expr_ would go here if needed,
        // but sticking to the attachment's apparent scope.
        _ => format!("<expr: {:?}>", expr), // Fallback for other Expr variants
    }
}

fn format_constant(constant: &Constant) -> String {
    match constant {
        Constant::Int(i) => i.to_string(),
        Constant::Float(f) => f.to_string(),
        Constant::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Constant::Str(s) => format!("\"{}\"", s.to_string().escape_default()),
        Constant::None => "None".to_string(),
        Constant::Ellipsis => "...".to_string(),
        Constant::Bytes(b) => format!("b\"{}\"", String::from_utf8_lossy(b).escape_default()),
        _ => format!("<const: {:?}>", constant),
    }
}

fn format_operator(op: &Operator) -> String {
    match op {
        Operator::Add => "+",
        Operator::Sub => "-",
        Operator::Mult => "*",
        Operator::MatMult => "@",
        Operator::Div => "/",
        Operator::Mod => "%",
        Operator::Pow => "**",
        Operator::LShift => "<<",
        Operator::RShift => ">>",
        Operator::BitOr => "|",
        Operator::BitXor => "^",
        Operator::BitAnd => "&",
        Operator::FloorDiv => "//",
    }
    .to_string()
}

fn format_unary_operator(op: &UnaryOp) -> String {
    match op {
        UnaryOp::Invert => "~",
        UnaryOp::Not => "not ",
        UnaryOp::UAdd => "+",
        UnaryOp::USub => "-",
    }
    .to_string()
}

fn format_bool_operator(op: &BoolOp) -> String {
    match op {
        BoolOp::And => "and",
        BoolOp::Or => "or",
    }
    .to_string()
}

fn format_cmp_operator(op: &CmpOp) -> String {
    match op {
        CmpOp::Eq => "==",
        CmpOp::NotEq => "!=",
        CmpOp::Lt => "<",
        CmpOp::LtE => "<=",
        CmpOp::Gt => ">",
        CmpOp::GtE => ">=",
        CmpOp::Is => "is",
        CmpOp::IsNot => "is not",
        CmpOp::In => "in",
        CmpOp::NotIn => "not in",
    }
    .to_string()
}

// format_stmt as per attachment structure, not using .node for matching
fn format_stmt(stmt: &Stmt) -> String {
    // Assuming Stmt is the enum itself, not Located<Stmt_> for this formatter's logic.
    // This matches the style of the provided attachment.
    match stmt {
        Stmt::Assign(StmtAssign {
            targets,
            value,
            type_comment: _, // Ignoring type_comment for simplicity
            ..
        }) => {
            let targets_str = targets
                .iter()
                .map(format_expr)
                .collect::<Vec<String>>()
                .join(", ");
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
            target, op, value, ..
        }) => {
            format!(
                "{} {}= {}",
                format_expr(target),
                format_operator(op),
                format_expr(value)
            )
        }
        // Stmt::Return and Stmt::Raise formatting within this function is not explicitly added
        // as per your "do not do this" instruction for format_stmt.
        // The Node::Return and Node::Raise in print_paths_to_writer handle their details.
        _ => format!("<stmt: {:?}>", stmt), // Fallback for other Stmt variants
    }
}

pub struct PathScraper;

impl PathScraper {
    pub fn get_paths(control_flow_graph: &ControlFlowGraph) -> Option<Vec<Vec<(NodeId, Edge)>>> {
        let paths = Self::traverse(
            control_flow_graph.get_graph(),
            control_flow_graph.get_entry(),
        );
        if paths.is_empty() {
            None
        } else {
            Some(paths)
        }
    }

    // Signature changed to include func_name
    pub fn print_paths_to_writer(
        func_name: &str,
        control_flow_graph: &ControlFlowGraph,
        writer: &mut dyn Write,
    ) -> std::io::Result<()> {
        // Print function name header
        writeln!(
            writer,
            "=== CONTROL FLOW GRAPH FOR FUNCTION: {} ===",
            func_name
        )?;

        // Print Nodes first, then Paths
        writeln!(writer, "=== CONTROL FLOW GRAPH NODES ===")?;
        let mut node_ids: Vec<_> = control_flow_graph.get_graph().keys().copied().collect();
        node_ids.sort();

        for node_id in &node_ids {
            if let Some(node) = control_flow_graph.get_node(*node_id) {
                writeln!(writer, "Node ID: {}", node_id)?;
                match node {
                    Node::Cond { stmts, expr, succ } => {
                        writeln!(writer, "  Type: Conditional")?;
                        if !stmts.is_empty() {
                            writeln!(writer, "  Stmts:")?;
                            for s in stmts {
                                writeln!(writer, "    - {}", format_stmt(s))?;
                            }
                        }
                        writeln!(writer, "  Condition: {}", format_expr(expr))?;
                        writeln!(
                            writer,
                            "  Successors: True -> {}, False -> {}",
                            succ[0], succ[1]
                        )?;
                    }
                    Node::Return { stmts, stmt } => {
                        writeln!(writer, "  Type: Return")?;
                        if !stmts.is_empty() {
                            writeln!(writer, "  Stmts:")?;
                            for s in stmts {
                                writeln!(writer, "    - {}", format_stmt(s))?;
                            }
                        }
                        // Details of the return statement (e.g., value) come from stmt directly
                        if let Some(value_expr) = &stmt.value {
                            // stmt is StmtReturn
                            writeln!(writer, "  Value: {}", format_expr(value_expr))?;
                        } else {
                            writeln!(writer, "  Value: None (implicit or explicit)")?;
                        }
                    }
                    Node::Raise { stmts, stmt } => {
                        writeln!(writer, "  Type: Raise")?;
                        if !stmts.is_empty() {
                            writeln!(writer, "  Stmts:")?;
                            for s in stmts {
                                writeln!(writer, "    - {}", format_stmt(s))?;
                            }
                        }
                        // Details of the raise statement (e.g., exception, cause) come from stmt directly
                        if let Some(exc_expr) = &stmt.exc {
                            // stmt is StmtRaise
                            writeln!(writer, "  Exception: {}", format_expr(exc_expr))?;
                        } else {
                            writeln!(writer, "  Exception: (bare raise)")?;
                        }
                        if let Some(cause_expr) = &stmt.cause {
                            writeln!(writer, "  Cause: {}", format_expr(cause_expr))?;
                        }
                    }
                }
            }
        }
        if !node_ids.is_empty() {
            writeln!(writer)?;
        }

        writeln!(writer, "=== CONTROL FLOW GRAPH PATHS ===")?;
        if let Some(all_paths) = Self::get_paths(control_flow_graph) {
            writeln!(writer, "Total paths found: {}", all_paths.len())?;
            if !all_paths.is_empty() {
                writeln!(writer)?;
            }

            for (i, path) in all_paths.iter().enumerate() {
                let path_str = path
                    .iter()
                    .enumerate()
                    .map(|(idx, (node_id, edge))| {
                        if idx < path.len() - 1 {
                            format!("Node {} --({:?})-->", node_id, edge)
                        } else {
                            if edge == &Edge::Terminal {
                                format!("Node {} (ends)", node_id)
                            } else {
                                format!(
                                    "Node {} --({:?})--> [Path did not end with TerminalEdge]",
                                    node_id, edge
                                )
                            }
                        }
                    })
                    .collect::<Vec<String>>()
                    .join(" ");
                writeln!(writer, "Path {}: {}", i, path_str)?;
            }
        } else {
            writeln!(writer, "No paths found in the control flow graph.")?;
        }
        writeln!(
            writer,
            "\n================================\n"
        )?; // Footer for function
        Ok(())
    }

    fn traverse(graph: &HashMap<NodeId, Node>, entry: NodeId) -> Vec<Vec<(NodeId, Edge)>> {
        let mut all_paths: Vec<Vec<(NodeId, Edge)>> = Vec::new();
        let mut stack: Vec<(NodeId, Vec<(NodeId, Edge)>)> = Vec::new();

        if graph.get(&entry).is_none() && entry == 0 && graph.is_empty() {
            return all_paths;
        }
        if graph.get(&entry).is_none() {
            return all_paths;
        }
        stack.push((entry, Vec::new()));

        while let Some((current_node_id, mut path_so_far)) = stack.pop() {
            if let Some(node) = graph.get(&current_node_id) {
                match node {
                    Node::Cond { succ, .. } => {
                        let mut false_path = path_so_far.clone();
                        false_path.push((current_node_id, Edge::False));
                        if graph.contains_key(&succ[1]) {
                            stack.push((succ[1], false_path));
                        }

                        path_so_far.push((current_node_id, Edge::True));
                        if graph.contains_key(&succ[0]) {
                            stack.push((succ[0], path_so_far));
                        }
                    }
                    Node::Return { .. } | Node::Raise { .. } => {
                        path_so_far.push((current_node_id, Edge::Terminal));
                        all_paths.push(path_so_far);
                    }
                }
            }
        }
        all_paths
    }
}
