use crate::cfg::{ControlFlowGraph, Edge, Node, NodeId};
use rustpython_ast::{
    BoolOp,
    CmpOp,
    Constant,
    Expr,
    ExprBoolOp,
    ExprCompare,
    ExprConstant,
    ExprContext, // Correct for rustpython-ast 0.4.0
    ExprName,
    ExprUnaryOp,
    Operator, // This is the correct type for BinOp.op and AugAssign.op
    Stmt,
    StmtAssert,
    StmtAssign,
    StmtAugAssign, // Make sure StmtAugAssign is imported
    StmtExpr,
    UnaryOp,
};
use std::collections::HashMap;
use z3::ast::{Ast, Bool, Dynamic, Int};
use z3::{Config, Context, SatResult, Solver};

use super::error::Z3Error;

/// Stores the result of a Z3 satisfiability check for a single path.
#[derive(Debug, Clone)]
pub struct PathConstraintResult {
    pub path_index: usize,
    pub is_satisfiable: bool,
    /// String representation of the Z3 model if satisfiable.
    pub model: Option<String>,
    /// Errors encountered during constraint generation or solving for this path.
    pub error: Option<Z3Error>,
}

/// Generates Z3 constraints from Python expressions.
///
/// This struct is responsible for translating Python AST expressions into
/// Z3 logical assertions. It maintains a map of Python variable names to
/// their Z3 counterparts to ensure consistency within a single path's analysis.
struct Z3ConstraintGenerator<'cfg> {
    z3_ctx: &'cfg Context,
    /// Maps Python variable names to Z3 AST nodes.
    /// Cleared for each new path to ensure variable independence.
    variable_map: HashMap<String, Dynamic<'cfg>>,
}

impl<'cfg> Z3ConstraintGenerator<'cfg> {
    /// Creates a new `Z3ConstraintGenerator` with the given Z3 context
    /// and pre-populates variables based on function arguments.
    fn new(
        ctx: &'cfg Context,
        function_args: &[(String, Option<String>)],
    ) -> Result<Self, Z3Error> {
        let mut variable_map = HashMap::new();
        for (name, type_hint_opt) in function_args {
            if let Some(type_hint) = type_hint_opt {
                match type_hint.as_str() {
                    "int" => {
                        let new_var = Int::new_const(ctx, name.to_string());
                        variable_map.insert(name.clone(), Dynamic::from_ast(&new_var));
                    }
                    "bool" => {
                        let new_var = Bool::new_const(ctx, name.to_string());
                        variable_map.insert(name.clone(), Dynamic::from_ast(&new_var));
                    }
                    // Add other type hints as needed, e.g., "str" -> z3::ast::String
                    _ => {
                        // If type hint is unknown or unhandled, don't pre-create.
                        // It will be created on first use with inferred type.
                    }
                }
            }
        }
        Ok(Z3ConstraintGenerator {
            z3_ctx: ctx,
            variable_map,
        })
    }

    /// Initializes the generator with function arguments, clearing any previous state.
    fn clear_and_reinitialize_args(
        &mut self,
        function_args: &[(String, Option<String>)],
    ) -> Result<(), Z3Error> {
        self.variable_map.clear();
        for (name, type_hint_opt) in function_args {
            if let Some(type_hint) = type_hint_opt {
                match type_hint.as_str() {
                    "int" => {
                        let new_var = Int::new_const(self.z3_ctx, name.to_string());
                        self.variable_map
                            .insert(name.clone(), Dynamic::from_ast(&new_var));
                    }
                    "bool" => {
                        let new_var = Bool::new_const(self.z3_ctx, name.to_string());
                        self.variable_map
                            .insert(name.clone(), Dynamic::from_ast(&new_var));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Retrieves an existing Z3 integer variable by name or creates a new one.
    fn get_or_create_int_var(&mut self, name: &str) -> Result<Int<'cfg>, Z3Error> {
        if let Some(ast_node) = self.variable_map.get(name) {
            return ast_node.as_int().ok_or_else(|| Z3Error::TypeMismatch {
                variable_name: name.to_string(),
                expected_type: "Int".to_string(),
            });
        }
        let new_var = Int::new_const(self.z3_ctx, name);
        self.variable_map
            .insert(name.to_string(), Dynamic::from_ast(&new_var));
        Ok(new_var)
    }

    /// Retrieves an existing Z3 boolean variable by name or creates a new one.
    fn get_or_create_bool_var(&mut self, name: &str) -> Result<Bool<'cfg>, Z3Error> {
        if let Some(ast_node) = self.variable_map.get(name) {
            return ast_node.as_bool().ok_or_else(|| Z3Error::TypeMismatch {
                variable_name: name.to_string(),
                expected_type: "Bool".to_string(),
            });
        }
        let new_var = Bool::new_const(self.z3_ctx, name);
        self.variable_map
            .insert(name.to_string(), Dynamic::from_ast(&new_var));
        Ok(new_var)
    }

    /// Converts a Python AST expression to a Z3 integer AST.
    fn python_expr_to_z3_int(&mut self, expr: &Expr) -> Result<Int<'cfg>, Z3Error> {
        match expr {
            Expr::Constant(ExprConstant { value, .. }) => match value {
                Constant::Int(i) => i
                    .try_into()
                    .map(|i64_val| Int::from_i64(self.z3_ctx, i64_val))
                    .map_err(|_| {
                        Z3Error::TypeConversion(format!(
                            "Integer value too large to convert to i64: {}",
                            i
                        ))
                    }),
                _ => Err(Z3Error::UnsupportedConstant {
                    value: value.clone(),
                    reason: "Expected an integer constant for Z3 integer conversion.".to_string(),
                }),
            },
            Expr::Name(ExprName { id, .. }) => self.get_or_create_int_var(id),
            _ => Err(Z3Error::UnsupportedExpressionType {
                expr_repr: format!("{:?}", expr),
            }),
        }
    }

    /// Converts a Python AST expression to a Z3 boolean AST.
    fn python_expr_to_z3_bool(&mut self, expr: &Expr) -> Result<Bool<'cfg>, Z3Error> {
        match expr {
            Expr::Constant(ExprConstant { value, .. }) => match value {
                Constant::Bool(b) => Ok(Bool::from_bool(self.z3_ctx, *b)),
                _ => Err(Z3Error::UnsupportedConstant {
                    value: value.clone(),
                    reason: "Expected a boolean constant (True/False) for Z3 boolean conversion."
                        .to_string(),
                }),
            },
            Expr::Name(ExprName { id, .. }) => self.get_or_create_bool_var(id),
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
                let mut z3_operands = Vec::with_capacity(values.len());
                for val_expr in values {
                    z3_operands.push(self.python_expr_to_z3_bool(val_expr)?);
                }
                let operands_ref: Vec<&Bool<'_>> = z3_operands.iter().collect();
                match op {
                    BoolOp::And => Ok(Bool::and(self.z3_ctx, &operands_ref)),
                    BoolOp::Or => Ok(Bool::or(self.z3_ctx, &operands_ref)),
                }
            }
            Expr::Compare(ExprCompare {
                left,
                ops,
                comparators,
                ..
            }) => {
                if ops.len() != 1 || comparators.len() != 1 {
                    return Err(Z3Error::ChainedComparisonNotSupported);
                }
                let z3_left = self.python_expr_to_z3_int(left)?;
                let z3_right = self.python_expr_to_z3_int(&comparators[0])?;
                match ops[0] {
                    CmpOp::Eq => Ok(z3_left._eq(&z3_right)),
                    CmpOp::NotEq => Ok(z3_left._eq(&z3_right).not()),
                    CmpOp::Lt => Ok(z3_left.lt(&z3_right)),
                    CmpOp::LtE => Ok(z3_left.le(&z3_right)),
                    CmpOp::Gt => Ok(z3_left.gt(&z3_right)),
                    CmpOp::GtE => Ok(z3_left.ge(&z3_right)),
                    _ => Err(Z3Error::UnsupportedCmpOperator { op: ops[0] }),
                }
            }
            _ => Err(Z3Error::UnsupportedExpressionType {
                expr_repr: format!("{:?}", expr),
            }),
        }
    }

    /// Converts a Python expression to a Z3 Dynamic AST.
    fn python_expr_to_z3_dynamic(&mut self, expr: &Expr) -> Result<Dynamic<'cfg>, Z3Error> {
        // Attempt to convert to Bool first
        if let Ok(b_val) = self.python_expr_to_z3_bool(expr) {
            return Ok(Dynamic::from_ast(&b_val));
        }
        // Then attempt to convert to Int
        if let Ok(i_val) = self.python_expr_to_z3_int(expr) {
            return Ok(Dynamic::from_ast(&i_val));
        }
        // Handle binary operations if not directly bool/int constants/names
        if let Expr::BinOp(bin_op_expr) = expr {
            let left_val = self.python_expr_to_z3_int(&bin_op_expr.left)?;
            let right_val = self.python_expr_to_z3_int(&bin_op_expr.right)?;
            match bin_op_expr.op {
                Operator::Add => {
                    return Ok(Dynamic::from_ast(&Int::add(
                        self.z3_ctx,
                        &[&left_val, &right_val],
                    )))
                }
                Operator::Sub => {
                    return Ok(Dynamic::from_ast(&Int::sub(
                        self.z3_ctx,
                        &[&left_val, &right_val],
                    )))
                }
                Operator::Mult => {
                    return Ok(Dynamic::from_ast(&Int::mul(
                        self.z3_ctx,
                        &[&left_val, &right_val],
                    )))
                }
                // Add other BinOp types as needed (Div, Mod, Pow, etc.)
                _ => {
                    return Err(Z3Error::UnsupportedExpressionType {
                        expr_repr: format!("Unsupported binary operator: {:?}", bin_op_expr.op),
                    })
                }
            }
        }
        Err(Z3Error::UnsupportedExpressionType {
            expr_repr: format!("Cannot convert to Z3 Dynamic: {:?}", expr),
        })
    }

    /// Processes a single Python statement and adds corresponding Z3 assertions.
    /// Updates the variable_map for assignments.
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

                            // Create a NEW Z3 variable for the LHS to represent its state *after* this assignment.
                            // The name for Z3 can be made unique, e.g., by appending a counter or using fresh_const.
                            // For simplicity with fresh_const, we can use a base name.
                            let new_lhs_z3_var = match z3_rhs_value.get_sort().kind() {
                                z3::SortKind::Bool => {
                                    Dynamic::from_ast(&Bool::fresh_const(self.z3_ctx, &format!("{}_val", target_name)))
                                }
                                z3::SortKind::Int => {
                                    Dynamic::from_ast(&Int::fresh_const(self.z3_ctx, &format!("{}_val", target_name)))
                                }
                                _ => {
                                    return Err(Z3Error::TypeConversion(
                                        "Unsupported Z3 sort for assignment target's new value".to_string(),
                                    ))
                                }
                            };

                            assertions.push(new_lhs_z3_var._eq(&z3_rhs_value));
                            // Update variable_map so 'target_name' now maps to this new Z3 variable.
                            self.variable_map.insert(target_name, new_lhs_z3_var);
                        }
                    }
                    // TODO: Handle other assignment targets (e.g., attributes, subscripts)
                }
                // TODO: Handle multiple assignment targets (e.g., a, b = 1, 2)
            }
            Stmt::AugAssign(StmtAugAssign { target, op, value, .. }) => {
                if let Expr::Name(ExprName { id, ctx, .. }) = target.as_ref() {
                    if matches!(ctx, ExprContext::Store) {
                        let target_name = id.to_string();

                        // 1. Evaluate current value of target (T_k) using current variable_map
                        let lhs_current_z3_val = self.python_expr_to_z3_dynamic(target)?;

                        // 2. Evaluate RHS value (V) using current variable_map
                        let rhs_z3_val = self.python_expr_to_z3_dynamic(value)?;

                        // 3. Compute result: Res_val = T_k op V
                        let result_val_z3_ast = match op {
                            Operator::Add => {
                                let l = lhs_current_z3_val.as_int().ok_or_else(|| Z3Error::TypeMismatch { variable_name: target_name.clone(), expected_type: "Int".to_string() })?;
                                let r = rhs_z3_val.as_int().ok_or_else(|| Z3Error::TypeMismatch { variable_name: "rhs".to_string(), expected_type: "Int".to_string() })?;
                                Int::add(self.z3_ctx, &[&l, &r])
                            }
                            Operator::Sub => {
                                let l = lhs_current_z3_val.as_int().ok_or_else(|| Z3Error::TypeMismatch { variable_name: target_name.clone(), expected_type: "Int".to_string() })?;
                                let r = rhs_z3_val.as_int().ok_or_else(|| Z3Error::TypeMismatch { variable_name: "rhs".to_string(), expected_type: "Int".to_string() })?;
                                Int::sub(self.z3_ctx, &[&l, &r])
                            }
                            // Add other Operator cases (Mult, Div, etc.) as needed
                            _ => return Err(Z3Error::UnsupportedExpressionType { expr_repr: format!("Unsupported AugAssign operator: {:?}", op) })
                        };
                        let result_val_dynamic = Dynamic::from_ast(&result_val_z3_ast);

                        // 4. Create new Z3 var for target's next state (T_k+1)
                        let new_lhs_z3_var = match result_val_dynamic.get_sort().kind() {
                             z3::SortKind::Bool => Dynamic::from_ast(&Bool::fresh_const(self.z3_ctx, &format!("{}_val_aug", target_name))),
                             z3::SortKind::Int => Dynamic::from_ast(&Int::fresh_const(self.z3_ctx, &format!("{}_val_aug", target_name))),
                             _ => return Err(Z3Error::TypeConversion("Unsupported Z3 sort for aug-assignment target's new value".to_string())),
                        };

                        // 5. Assert T_k+1 == Res_val
                        assertions.push(new_lhs_z3_var._eq(&result_val_dynamic));

                        // 6. Update map: target_name -> T_k+1
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
            Stmt::Pass(_) => {
                // No Z3 assertion for pass
            }
            _ => {
                // Optionally log or error for unsupported statements
            }
        }
        Ok(())
    }

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

            let stmts_to_process: &Vec<Stmt>;
            let mut condition_expr_opt: Option<&Expr> = None;

            match node {
                Node::Cond { stmts, expr, .. } => {
                    stmts_to_process = stmts;
                    condition_expr_opt = Some(expr);
                }
                Node::Return { stmts, .. } | Node::Raise { stmts, .. } => {
                    stmts_to_process = stmts;
                }
            }

            for stmt in stmts_to_process {
                self.process_statement_for_z3(stmt, &mut path_assertions)?;
            }

            if let Some(condition_expr) = condition_expr_opt {
                if let Node::Cond { .. } = node {
                    // Ensure it's actually a Cond node for edge processing
                    let condition_ast = self.python_expr_to_z3_bool(condition_expr)?;
                    match edge {
                        Edge::True => path_assertions.push(condition_ast),
                        Edge::False => path_assertions.push(condition_ast.not()),
                        Edge::Terminal => { /* Terminal edge on Cond node is unusual, implies path ends mid-condition */
                        }
                    }
                }
            }
            // For Return/Raise nodes, their `stmts` are processed, and the path ends.
            // The `Edge::Terminal` itself doesn't add a Z3 condition from the node's expression.
        }

        if path_assertions.is_empty() {
            Ok(Bool::from_bool(self.z3_ctx, true))
        } else {
            Ok(Bool::and(
                self.z3_ctx,
                &path_assertions.iter().collect::<Vec<_>>(),
            ))
        }
    }
}

/// Analyzes all paths from the CFG, generates Z3 constraints for each,
/// and checks their satisfiability.
pub fn analyze_paths(
    all_paths: &Vec<Vec<(NodeId, Edge)>>,
    cfg_data: &ControlFlowGraph,
) -> Vec<PathConstraintResult> {
    let z3_config = Config::new();
    let z3_ctx = Context::new(&z3_config);
    // Initialize with function arguments
    let mut constraint_generator =
        match Z3ConstraintGenerator::new(&z3_ctx, cfg_data.get_arguments()) {
            Ok(gen) => gen,
            Err(e) => {
                // If generator creation fails, return results with this error for all paths.
                // This is a simplification; ideally, we'd have a way to signal this top-level error.
                return all_paths
                    .iter()
                    .enumerate()
                    .map(|(i, _)| PathConstraintResult {
                        path_index: i,
                        is_satisfiable: false,
                        model: None,
                        error: Some(e.clone()), // Clone the error for each path result
                    })
                    .collect();
            }
        };
    let mut results = Vec::new();

    for (i, path) in all_paths.iter().enumerate() {
        // Re-initialize argument types in the variable_map for the current path analysis
        // This ensures that even if other variables were added, args have priority for type.
        if constraint_generator
            .clear_and_reinitialize_args(cfg_data.get_arguments())
            .is_err()
        {
            // Handle error during re-initialization if necessary, though current impl always Ok
            results.push(PathConstraintResult {
                path_index: i,
                is_satisfiable: false,
                model: None,
                error: Some(Z3Error::TypeConversion(
                    "Failed to reinitialize arguments".to_string(),
                )),
            });
            continue;
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
                            solver
                                .get_reason_unknown()
                                .unwrap_or_else(|| "Reason unknown".to_string()),
                        ));
                    }
                }
            }
            Err(e) => {
                path_result.error = Some(e);
            }
        }
        results.push(path_result);
    }
    results
}

/// Example of how you might call `analyze_paths`.
#[allow(dead_code)]
pub fn print_paths(cfg: &ControlFlowGraph, paths: &Vec<Vec<(NodeId, Edge)>>) {
    let constraint_results = analyze_paths(paths, cfg);

    println!("=== PATH ANALYSIS RESULTS ===");
    println!("Total paths found: {}\n", constraint_results.len());

    for result in constraint_results {
        println!(
            "Path {}: Satisfiable = {}",
            result.path_index, result.is_satisfiable
        );

        // Print just the path as (NodeId, Edge) pairs
        if let Some(path) = paths.get(result.path_index) {
            println!("  Path: {:?}", path);
        }

        // Print the Z3 model if satisfiable
        if let Some(model_str) = &result.model {
            println!("  Z3 Model:");
            for line in model_str.lines() {
                if !line.trim().is_empty() {
                    println!("    {}", line.trim());
                }
            }
        }

        // Print any errors
        if let Some(err) = &result.error {
            println!("  Error: {}", err);
        }

        println!(); // Empty line between paths
    }
}
