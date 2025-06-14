use crate::cfg::{ControlFlowGraph, Edge, Node, NodeId};
use rustpython_ast::{
    BoolOp,
    CmpOp,
    Constant,
    Expr,
    ExprBinOp, 
    ExprBoolOp,
    ExprCompare,
    ExprConstant,
    ExprContext,
    ExprName,
    ExprUnaryOp,
    Operator,
    Stmt,
    StmtAssert,
    StmtAssign,
    StmtAugAssign,
    StmtExpr,
    StmtPass, 
    UnaryOp,
};
use std::collections::HashMap;
// Import Z3 String type and alias it to avoid conflict with std::string::String
use z3::ast::{Ast, Bool, Dynamic, Int, String as Z3String};
use z3::{Config, Context, SatResult, Solver};

use super::error::Z3Error;

/// Stores the result of a Z3 satisfiability check for a single path.
#[derive(Debug, Clone)]
pub struct PathConstraintResult {
    /// Index of the path in the original list of all paths.
    pub path_index: usize,
    /// Whether the path constraints were satisfiable.
    pub is_satisfiable: bool,
    /// String representation of the Z3 model if satisfiable.
    pub model: Option<String>,
    /// Errors encountered during constraint generation or solving for this path.
    pub error: Option<Z3Error>,
}

/// Generates Z3 constraints from Python expressions and statements along a path.
///
/// This struct translates Python AST elements into Z3 logical assertions.
/// It maintains a map of Python variable names to their Z3 counterparts,
/// creating new Z3 variables for assignments to model state changes (SSA-like).
struct Z3ConstraintGenerator<'cfg> {
    z3_ctx: &'cfg Context,
    /// Maps Python variable names to their current Z3 AST `Dynamic` representation.
    /// This map is updated as assignments are processed along a path.
    variable_map: HashMap<String, Dynamic<'cfg>>,
}

impl<'cfg> Z3ConstraintGenerator<'cfg> {
    /// Creates a new `Z3ConstraintGenerator`.
    /// Initializes `variable_map` with Z3 variables for function arguments based on type hints.
    fn new(
        ctx: &'cfg Context,
        function_args: &[(String, Option<String>)],
    ) -> Result<Self, Z3Error> {
        let mut variable_map = HashMap::new();
        for (name, type_hint_opt) in function_args {
            if let Some(type_hint) = type_hint_opt {
                match type_hint.as_str() {
                    "int" => {
                        let new_var = Int::new_const(ctx, name.as_str());
                        variable_map.insert(name.clone(), Dynamic::from_ast(&new_var));
                    }
                    "bool" => {
                        let new_var = Bool::new_const(ctx, name.as_str());
                        variable_map.insert(name.clone(), Dynamic::from_ast(&new_var));
                    }
                    "str" => { // Added string type hint handling
                        let new_var = Z3String::new_const(ctx, name.as_str());
                        variable_map.insert(name.clone(), Dynamic::from_ast(&new_var));
                    }
                    _ => {}
                }
            }
        }
        Ok(Z3ConstraintGenerator {
            z3_ctx: ctx,
            variable_map,
        })
    }

    /// Clears `variable_map` and re-initializes it with Z3 variables for function arguments.
    /// This is called at the start of processing each new path.
    fn clear_and_reinitialize_args(
        &mut self,
        function_args: &[(String, Option<String>)],
    ) -> Result<(), Z3Error> {
        self.variable_map.clear();
        for (name, type_hint_opt) in function_args {
            if let Some(type_hint) = type_hint_opt {
                match type_hint.as_str() {
                    "int" => {
                        let new_var = Int::new_const(self.z3_ctx, name.as_str());
                        self.variable_map
                            .insert(name.clone(), Dynamic::from_ast(&new_var));
                    }
                    "bool" => {
                        let new_var = Bool::new_const(self.z3_ctx, name.as_str());
                        self.variable_map
                            .insert(name.clone(), Dynamic::from_ast(&new_var));
                    }
                    "str" => { // Added string type hint handling
                        let new_var = Z3String::new_const(self.z3_ctx, name.as_str());
                        self.variable_map
                            .insert(name.clone(), Dynamic::from_ast(&new_var));
                    }
                    _ => {} 
                }
            }
        }
        Ok(())
    }

    /// Retrieves an existing Z3 integer variable or creates a new one if not found.
    fn get_or_create_int_var(&mut self, name: &str) -> Result<Int<'cfg>, Z3Error> {
        if let Some(ast_node) = self.variable_map.get(name) {
            ast_node.as_int().ok_or_else(|| Z3Error::TypeMismatch {
                variable_name: name.to_string(),
                expected_type: "Int".to_string(),
            })
        } else {
            let new_var = Int::new_const(self.z3_ctx, name);
            self.variable_map.insert(name.to_string(), Dynamic::from_ast(&new_var));
            Ok(new_var)
        }
    }
    
    /// Retrieves an existing Z3 boolean variable or creates a new one if not found.
    fn get_or_create_bool_var(&mut self, name: &str) -> Result<Bool<'cfg>, Z3Error> {
        if let Some(ast_node) = self.variable_map.get(name) {
            ast_node.as_bool().ok_or_else(|| Z3Error::TypeMismatch {
                variable_name: name.to_string(),
                expected_type: "Bool".to_string(),
            })
        } else {
            let new_var = Bool::new_const(self.z3_ctx, name);
            self.variable_map.insert(name.to_string(), Dynamic::from_ast(&new_var));
            Ok(new_var)
        }
    }

    /// Retrieves an existing Z3 string variable or creates a new one if not found.
    fn get_or_create_string_var(&mut self, name: &str) -> Result<Z3String<'cfg>, Z3Error> {
        if let Some(ast_node) = self.variable_map.get(name) {
            ast_node.as_string().ok_or_else(|| Z3Error::TypeMismatch {
                variable_name: name.to_string(),
                expected_type: "String".to_string(),
            })
        } else {
            let new_var = Z3String::new_const(self.z3_ctx, name);
            self.variable_map.insert(name.to_string(), Dynamic::from_ast(&new_var));
            Ok(new_var)
        }
    }

    /// Converts a Python AST `Expr` to a Z3 `Int` AST.
    fn python_expr_to_z3_int(&mut self, expr: &Expr) -> Result<Int<'cfg>, Z3Error> {
        match expr {
            Expr::Constant(ExprConstant { value, .. }) => match value {
                Constant::Int(i) => {
                    // Try to convert num_bigint::BigInt to i64 for Z3.
                    // This might lose precision for very large numbers not fitting in i64.
                    let i64_val = i.try_into().map_err(|_| {
                        Z3Error::TypeConversion(format!("Integer value {} too large for i64", i))
                    })?;
                    Ok(Int::from_i64(self.z3_ctx, i64_val))
                }
                _ => Err(Z3Error::UnsupportedConstant {
                    value: value.clone(),
                    reason: "Expected an integer constant for Z3 Int conversion.".to_string(),
                }),
            },
            Expr::Name(ExprName { id, .. }) => self.get_or_create_int_var(id.as_str()),
            _ => Err(Z3Error::UnsupportedExpressionType {
                expr_repr: format!("{:?}", expr),
            }),
        }
    }

    /// Converts a Python AST `Expr` to a Z3 `String` AST.
    fn python_expr_to_z3_string(&mut self, expr: &Expr) -> Result<Z3String<'cfg>, Z3Error> {
        match expr {
            Expr::Constant(ExprConstant { value, .. }) => match value {
                Constant::Str(s) => Z3String::from_str(self.z3_ctx, s.as_str())
                    .map_err(|e| Z3Error::TypeConversion(format!("Failed to create Z3 string: {}", e))),
                _ => Err(Z3Error::UnsupportedConstant {
                    value: value.clone(),
                    reason: "Expected a string constant for Z3 String conversion.".to_string(),
                }),
            },
            Expr::Name(ExprName { id, .. }) => self.get_or_create_string_var(id.as_str()),
            _ => Err(Z3Error::UnsupportedExpressionType {
                expr_repr: format!("Cannot convert to Z3 String: {:?}", expr),
            }),
        }
    }
    
    /// Converts a Python AST `Expr` to a Z3 `Bool` AST.
    fn python_expr_to_z3_bool(&mut self, expr: &Expr) -> Result<Bool<'cfg>, Z3Error> {
        match expr {
            Expr::Constant(ExprConstant { value, .. }) => match value {
                Constant::Bool(b) => Ok(Bool::from_bool(self.z3_ctx, *b)),
                _ => Err(Z3Error::UnsupportedConstant {
                    value: value.clone(),
                    reason: "Expected True/False for Z3 Bool conversion.".to_string(),
                }),
            },
            Expr::Name(ExprName { id, .. }) => self.get_or_create_bool_var(id.as_str()),
            Expr::UnaryOp(ExprUnaryOp { op, operand, .. }) => {
                let z3_operand = self.python_expr_to_z3_bool(operand)?;
                match op {
                    UnaryOp::Not => Ok(z3_operand.not()),
                    _ => Err(Z3Error::UnsupportedUnaryOperator { op: *op }),
                }
            }
            Expr::BoolOp(ExprBoolOp { op, values, .. }) => {
                if values.is_empty() {
                    return Err(Z3Error::EmptyBoolOpValues);
                }
                // Convert all operand expressions to Z3 Bool.
                let z3_operands: Result<Vec<Bool<'cfg>>, Z3Error> = values
                    .iter()
                    .map(|val_expr| self.python_expr_to_z3_bool(val_expr))
                    .collect();
                let z3_operands = z3_operands?; // Propagate error if any conversion failed.

                // Z3 and/or require a slice of references.
                let operands_ref: Vec<&Bool<'_>> = z3_operands.iter().collect();
                match op {
                    BoolOp::And => Ok(Bool::and(self.z3_ctx, &operands_ref)),
                    BoolOp::Or => Ok(Bool::or(self.z3_ctx, &operands_ref)),
                }
            }
            Expr::Compare(ExprCompare { left, ops, comparators, .. }) => {
                // Refactored to handle comparisons based on dynamic types.
                if ops.len() != 1 || comparators.len() != 1 {
                    return Err(Z3Error::ChainedComparisonNotSupported);
                }

                let z3_left_dynamic = self.python_expr_to_z3_dynamic(left)?;
                let z3_right_dynamic = self.python_expr_to_z3_dynamic(&comparators[0])?;
                let op = ops[0];

                match (z3_left_dynamic.get_sort().kind(), z3_right_dynamic.get_sort().kind()) {
                    (z3::SortKind::Int, z3::SortKind::Int) => {
                        let l = z3_left_dynamic.as_int().unwrap(); 
                        let r = z3_right_dynamic.as_int().unwrap();
                        match op {
                            CmpOp::Eq => Ok(l._eq(&r)),
                            CmpOp::NotEq => Ok(l._eq(&r).not()),
                            CmpOp::Lt => Ok(l.lt(&r)),
                            CmpOp::LtE => Ok(l.le(&r)),
                            CmpOp::Gt => Ok(l.gt(&r)),
                            CmpOp::GtE => Ok(l.ge(&r)),
                            _ => Err(Z3Error::UnsupportedCmpOperatorForSort { op, sort_name: "Int".to_string() }),
                        }
                    }
                    (z3::SortKind::Bool, z3::SortKind::Bool) => {
                        let l = z3_left_dynamic.as_bool().unwrap();
                        let r = z3_right_dynamic.as_bool().unwrap();
                        match op {
                            CmpOp::Eq => Ok(l._eq(&r)),
                            CmpOp::NotEq => Ok(l._eq(&r).not()),
                            _ => Err(Z3Error::UnsupportedCmpOperatorForSort { op, sort_name: "Bool".to_string() }),
                        }
                    }
                    (z3::SortKind::Seq, z3::SortKind::Seq) => {
                        let l = z3_left_dynamic.as_string().unwrap();
                        let r = z3_right_dynamic.as_string().unwrap();
                        match op {
                            CmpOp::Eq => Ok(l._eq(&r)),
                            CmpOp::NotEq => Ok(l._eq(&r).not()),
                            _ => Err(Z3Error::UnsupportedCmpOperatorForSort { op, sort_name: "String".to_string() }),
                        }
                    }
                    (left_kind, right_kind) => Err(Z3Error::TypeMismatchInComparison {
                        op,
                        left_type: format!("{:?}", left_kind),
                        right_type: format!("{:?}", right_kind),
                    }),
                }
            }
            _ => Err(Z3Error::UnsupportedExpressionType {
                expr_repr: format!("{:?}", expr),
            }),
        }
    }

    /// Converts a Python `Expr` to a Z3 `Dynamic` AST.
    fn python_expr_to_z3_dynamic(&mut self, expr: &Expr) -> Result<Dynamic<'cfg>, Z3Error> {
        // Order of attempts: String, Bool, then Int.
        if let Ok(s_val) = self.python_expr_to_z3_string(expr) {
            return Ok(Dynamic::from_ast(&s_val));
        }
        if let Ok(b_val) = self.python_expr_to_z3_bool(expr) {
            return Ok(Dynamic::from_ast(&b_val));
        }
        if let Ok(i_val) = self.python_expr_to_z3_int(expr) {
            return Ok(Dynamic::from_ast(&i_val));
        }
        
        if let Expr::BinOp(ExprBinOp { left, op, right, .. }) = expr {
            // Current BinOp support is primarily for Int. String concatenation would go here.
            let left_val = self.python_expr_to_z3_int(left)?;
            let right_val = self.python_expr_to_z3_int(right)?;
            match op {
                Operator::Add => Ok(Dynamic::from_ast(&Int::add(self.z3_ctx, &[&left_val, &right_val]))),
                Operator::Sub => Ok(Dynamic::from_ast(&Int::sub(self.z3_ctx, &[&left_val, &right_val]))),
                Operator::Mult => Ok(Dynamic::from_ast(&Int::mul(self.z3_ctx, &[&left_val, &right_val]))),
                _ => Err(Z3Error::UnsupportedExpressionType {
                    expr_repr: format!("Unsupported binary operator: {:?}", op),
                }),
            }
        } else {
            Err(Z3Error::UnsupportedExpressionType {
                expr_repr: format!("Cannot convert to Z3 Dynamic: {:?}", expr),
            })
        }
    }

    /// Processes a single Python `Stmt` to generate Z3 assertions and update `variable_map`.
    fn process_statement_for_z3(
        &mut self,
        stmt: &Stmt,
        assertions: &mut Vec<Bool<'cfg>>,
    ) -> Result<(), Z3Error> {
        match stmt {
            Stmt::Assign(StmtAssign { targets, value, .. }) => {
                if targets.len() == 1 {
                    if let Expr::Name(ExprName { id, ctx, .. }) = &targets[0] {
                        if matches!(ctx, ExprContext::Store) {
                            let target_name = id.to_string();
                            let z3_rhs_value = self.python_expr_to_z3_dynamic(value)?;

                            let new_lhs_z3_var = match z3_rhs_value.get_sort().kind() {
                                z3::SortKind::Bool => Dynamic::from_ast(&Bool::fresh_const(self.z3_ctx, &format!("{}_assigned_bool", target_name))),
                                z3::SortKind::Int => Dynamic::from_ast(&Int::fresh_const(self.z3_ctx, &format!("{}_assigned_int", target_name))),
                                z3::SortKind::Seq => Dynamic::from_ast(&Z3String::fresh_const(self.z3_ctx, &format!("{}_assigned_str", target_name))), // Added String
                                _ => return Err(Z3Error::TypeConversion("Unsupported Z3 sort for assignment target's new value".to_string())),
                            };

                            assertions.push(new_lhs_z3_var._eq(&z3_rhs_value));
                            self.variable_map.insert(target_name, new_lhs_z3_var);
                        }
                    }
                }
            }
            Stmt::AugAssign(StmtAugAssign { target, op, value, .. }) => {
                 if let Expr::Name(ExprName { id, ctx, .. }) = target.as_ref() { 
                    if matches!(ctx, ExprContext::Store) { 
                        let target_name = id.to_string();
                        let lhs_current_z3_val = self.python_expr_to_z3_dynamic(target)?;
                        let rhs_z3_val = self.python_expr_to_z3_dynamic(value)?;

                        // AugAssign for strings (e.g. += for concat) is not yet supported here.
                        // This section primarily handles numeric AugAssign.
                        let lhs_int = lhs_current_z3_val.as_int().ok_or_else(|| Z3Error::TypeMismatch { variable_name: target_name.clone(), expected_type: "Int for AugAssign".to_string() })?;
                        let rhs_int = rhs_z3_val.as_int().ok_or_else(|| Z3Error::TypeMismatch { variable_name: "RHS of AugAssign".to_string(), expected_type: "Int for AugAssign".to_string() })?;
                        
                        let result_val_ast = match op {
                            Operator::Add => Int::add(self.z3_ctx, &[&lhs_int, &rhs_int]),
                            Operator::Sub => Int::sub(self.z3_ctx, &[&lhs_int, &rhs_int]),
                            Operator::Mult => Int::mul(self.z3_ctx, &[&lhs_int, &rhs_int]),
                            _ => return Err(Z3Error::UnsupportedExpressionType { expr_repr: format!("Unsupported AugAssign operator: {:?}", op) })
                        };
                        let result_val_dynamic = Dynamic::from_ast(&result_val_ast);

                        let new_lhs_z3_var = match result_val_dynamic.get_sort().kind() {
                             z3::SortKind::Bool => Dynamic::from_ast(&Bool::fresh_const(self.z3_ctx, &format!("{}_aug_assigned_bool", target_name))), // Should not happen if result is Int
                             z3::SortKind::Int => Dynamic::from_ast(&Int::fresh_const(self.z3_ctx, &format!("{}_aug_assigned_int", target_name))),
                             // String case for AugAssign would be needed if supporting string concatenation
                             // z3::SortKind::Seq => Dynamic::from_ast(&Z3String::fresh_const(self.z3_ctx, &format!("{}_aug_assigned_str", target_name))),
                             _ => return Err(Z3Error::TypeConversion("Unsupported Z3 sort for aug-assignment target's new value".to_string())),
                        };

                        assertions.push(new_lhs_z3_var._eq(&result_val_dynamic));
                        self.variable_map.insert(target_name, new_lhs_z3_var);
                    }
                }
            }
            Stmt::Expr(StmtExpr { value, .. }) => {
                if let Expr::Call(call_expr) = value.as_ref() { 
                    if let Expr::Name(name_expr) = call_expr.func.as_ref() { 
                        if name_expr.id.as_str() == "assert" && !call_expr.args.is_empty() {
                            let condition = self.python_expr_to_z3_bool(&call_expr.args[0])?;
                            assertions.push(condition);
                        }
                    }
                }
            }
            Stmt::Assert(StmtAssert { test, .. }) => { 
                let condition = self.python_expr_to_z3_bool(test)?;
                assertions.push(condition);
            }
            Stmt::Pass(_) => {}
            _ => {}
        }
        Ok(())
    }

    /// Creates a single Z3 `Bool` AST representing all constraints for a given path.
    fn create_path_assertion(
        &mut self,
        path: &[(NodeId, Edge)],
        cfg_data: &ControlFlowGraph,
    ) -> Result<Bool<'cfg>, Z3Error> {
        let mut path_assertions: Vec<Bool<'cfg>> = Vec::new();

        for (node_id, edge) in path {
            let node = cfg_data
                .get_node(*node_id)
                .ok_or(Z3Error::NodeNotFoundInCfg(*node_id))?;

            // Process statements within the block leading to this node.
            let stmts_to_process = match node {
                Node::Cond { stmts, .. } | Node::Return { stmts, .. } | Node::Raise { stmts, .. } => stmts,
            };
            for stmt in stmts_to_process {
                self.process_statement_for_z3(stmt, &mut path_assertions)?;
            }

            // Add constraint for the conditional expression if this is a Cond node.
            if let Node::Cond { expr: condition_expr, .. } = node {
                let condition_ast = self.python_expr_to_z3_bool(condition_expr)?;
                match edge {
                    Edge::True => path_assertions.push(condition_ast),
                    Edge::False => path_assertions.push(condition_ast.not()),
                    Edge::Terminal => {
                        // This case should ideally not occur for a Cond node's edge in a valid path.
                        // A Terminal edge implies the path ends, usually at a Return/Raise node.
                        // If a Cond node is followed by a Terminal edge, it might indicate an issue
                        // in path scraping or CFG structure.
                        // For now, we don't add a constraint for it.
                    }
                }
            }
            // For Return/Raise nodes, their `stmts` are processed. The `Edge::Terminal` itself
            // doesn't add a new Z3 condition based on the node's *expression*, as they don't have one
            // in the same way Cond nodes do. The path simply terminates.
        }

        // Combine all assertions for the path. If no assertions, it's a trivially true path.
        if path_assertions.is_empty() {
            Ok(Bool::from_bool(self.z3_ctx, true))
        } else {
            Ok(Bool::and(self.z3_ctx, &path_assertions.iter().collect::<Vec<_>>()))
        }
    }
}

/// Analyzes all paths from the CFG, generates Z3 constraints, and checks satisfiability.
pub fn analyze_paths(
    all_paths: &[Vec<(NodeId, Edge)>], 
    cfg_data: &ControlFlowGraph,
) -> Vec<PathConstraintResult> {
    let z3_config = Config::new();
    let z3_ctx = Context::new(&z3_config);

    // Initialize constraint generator.
    // If this fails, all paths will be marked with this initialization error.
    let mut constraint_generator = match Z3ConstraintGenerator::new(&z3_ctx, cfg_data.get_arguments()) {
        Ok(gen) => gen,
        Err(e) => {
            return all_paths
                .iter()
                .enumerate()
                .map(|(i, _)| PathConstraintResult {
                    path_index: i,
                    is_satisfiable: false,
                    model: None,
                    error: Some(e.clone()),
                })
                .collect();
        }
    };

    // Use `map` and `collect` for a more functional approach to processing paths.
    all_paths
        .iter()
        .enumerate()
        .map(|(i, path)| {
            // Re-initialize argument types in the variable_map for each path.
            // This ensures a clean state for each path's analysis.
            if let Err(reinit_err) = constraint_generator.clear_and_reinitialize_args(cfg_data.get_arguments()) {
                return PathConstraintResult {
                    path_index: i,
                    is_satisfiable: false,
                    model: None,
                    error: Some(reinit_err), 
                };
            }

            let solver = Solver::new(&z3_ctx);
            let mut path_result = PathConstraintResult {
                path_index: i,
                is_satisfiable: false,
                model: None,
                error: None,
            };

            match constraint_generator.create_path_assertion(path, cfg_data) {
                Ok(path_assertion) => {
                    solver.assert(&path_assertion);
                    match solver.check() {
                        SatResult::Sat => {
                            path_result.is_satisfiable = true;
                            if let Some(m) = solver.get_model() {
                                path_result.model = Some(m.to_string());
                            }
                        }
                        SatResult::Unsat => {
                            path_result.is_satisfiable = false;
                        }
                        SatResult::Unknown => {
                            path_result.is_satisfiable = false;
                            path_result.error = Some(Z3Error::SolverUnknown(
                                solver.get_reason_unknown().unwrap_or_else(|| "Reason unknown".to_string()),
                            ));
                        }
                    }
                }
                Err(e) => {
                    path_result.error = Some(e);
                }
            }
            path_result
        })
        .collect()
}

/// Prints path analysis results, including Z3 models and errors.
/// (This function is primarily for debugging and demonstration).
#[allow(dead_code)] // Allow dead code as this is a utility/debug function.
pub fn print_paths_analysis(cfg: &ControlFlowGraph, paths: &[Vec<(NodeId, Edge)>]) { 
    let constraint_results = analyze_paths(paths, cfg);

    println!("=== PATH ANALYSIS RESULTS ===");
    println!("Total paths analyzed: {}\n", constraint_results.len());

    for result in constraint_results {
        println!(
            "Path {}: Satisfiable = {}",
            result.path_index, result.is_satisfiable
        );

        // Print the path structure.
        if let Some(path_structure) = paths.get(result.path_index) {
            println!("  Path Structure: {:?}", path_structure);
        }

        // Print the Z3 model if satisfiable and available.
        if let Some(model_str) = &result.model {
            println!("  Z3 Model:");
            // Filter out empty lines from model string for cleaner output.
            model_str.lines().filter(|line| !line.trim().is_empty()).for_each(|line| {
                println!("    {}", line.trim());
            });
        }

        // Print any errors encountered.
        if let Some(err) = &result.error {
            println!("  Error: {}", err);
        }
        println!(); // Add a blank line for readability between path results.
    }
}
