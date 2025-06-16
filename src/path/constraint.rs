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
    UnaryOp, // Ensure UnaryOp is imported
};
use std::collections::HashMap;
// Import Z3 Real type
use z3::ast::{Ast, Bool, Dynamic, Int, Real, String as Z3String};
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
    variable_map: HashMap<String, Dynamic<'cfg>>,
}

/// Creates a Z3 variable of a specific type if it doesn't exist in the variable_map,
/// or retrieves and casts it if it does.
macro_rules! get_or_create_z3_var {
    ($self:ident, $name:expr, $z3_type:ty, $constructor_fn:path, $caster_method:ident, $type_name_str:expr) => {
        if let Some(ast_node) = $self.variable_map.get($name) {
            ast_node
                .$caster_method()
                .ok_or_else(|| Z3Error::TypeMismatch {
                    variable_name: $name.to_string(),
                    expected_type: $type_name_str.to_string(),
                })
        } else {
            let new_var = $constructor_fn($self.z3_ctx, $name);
            $self
                .variable_map
                .insert($name.to_string(), Dynamic::from_ast(&new_var));
            Ok(new_var)
        }
    };
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
                    "str" => {
                        let new_var = Z3String::new_const(ctx, name.as_str());
                        variable_map.insert(name.clone(), Dynamic::from_ast(&new_var));
                    }
                    "float" => {
                        // Added float handling
                        let new_var = Real::new_const(ctx, name.as_str());
                        variable_map.insert(name.clone(), Dynamic::from_ast(&new_var));
                    }
                    _ => {} // Unknown type hints are ignored for now
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
                    "str" => {
                        let new_var = Z3String::new_const(self.z3_ctx, name.as_str());
                        self.variable_map
                            .insert(name.clone(), Dynamic::from_ast(&new_var));
                    }
                    "float" => {
                        // Added float handling
                        let new_var = Real::new_const(self.z3_ctx, name.as_str());
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
        get_or_create_z3_var!(self, name, Int<'cfg>, Int::new_const, as_int, "Int")
    }

    /// Retrieves an existing Z3 boolean variable or creates a new one if not found.
    fn get_or_create_bool_var(&mut self, name: &str) -> Result<Bool<'cfg>, Z3Error> {
        get_or_create_z3_var!(self, name, Bool<'cfg>, Bool::new_const, as_bool, "Bool")
    }

    /// Retrieves an existing Z3 string variable or creates a new one if not found.
    fn get_or_create_string_var(&mut self, name: &str) -> Result<Z3String<'cfg>, Z3Error> {
        get_or_create_z3_var!(
            self,
            name,
            Z3String<'cfg>,
            Z3String::new_const,
            as_string,
            "String"
        )
    }

    /// Retrieves an existing Z3 real variable or creates a new one if not found.
    fn get_or_create_real_var(&mut self, name: &str) -> Result<Real<'cfg>, Z3Error> {
        get_or_create_z3_var!(self, name, Real<'cfg>, Real::new_const, as_real, "Real")
    }

    /// Converts a Python AST `Expr` to a Z3 `Int` AST.
    fn python_expr_to_z3_int(&mut self, expr: &Expr) -> Result<Int<'cfg>, Z3Error> {
        match &expr {
            // Changed to match &expr
            Expr::Constant(ExprConstant { value, .. }) => match value {
                Constant::Int(i) => {
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
                expr_repr: format!("{:?}", expr), // Use expr
            }),
        }
    }

    /// Converts a Python AST `Expr` to a Z3 `String` AST.
    fn python_expr_to_z3_string(&mut self, expr: &Expr) -> Result<Z3String<'cfg>, Z3Error> {
        match &expr {
            // Changed to match &expr
            Expr::Constant(ExprConstant { value, .. }) => match value {
                Constant::Str(s) => Z3String::from_str(self.z3_ctx, s.as_str()).map_err(|e| {
                    Z3Error::TypeConversion(format!("Failed to create Z3 string: {}", e))
                }),
                _ => Err(Z3Error::UnsupportedConstant {
                    value: value.clone(),
                    reason: "Expected a string constant for Z3 String conversion.".to_string(),
                }),
            },
            Expr::Name(ExprName { id, .. }) => self.get_or_create_string_var(id.as_str()),
            _ => Err(Z3Error::UnsupportedExpressionType {
                expr_repr: format!("Cannot convert to Z3 String: {:?}", expr), // Use expr
            }),
        }
    }

    /// Converts a Python AST `Expr` to a Z3 `Real` AST.
    fn python_expr_to_z3_real(&mut self, expr: &Expr) -> Result<Real<'cfg>, Z3Error> {
        match &expr {
            // Changed to match &expr
            Expr::Constant(ExprConstant { value, .. }) => match value {
                Constant::Float(f) => {
                    // Z3 Real::from_real expects a numerator and denominator
                    // For floats, we'll use the string representation and convert to rational
                    Ok(Real::from_real(
                        self.z3_ctx,
                        f.to_string().parse::<i32>().unwrap_or(0),
                        1,
                    ))
                }
                Constant::Int(i) => {
                    // Allow int to be converted to real
                    let i_val: i32 = i.try_into().map_err(|_| {
                        Z3Error::TypeConversion(format!("Integer value {} too large for i32", i))
                    })?;
                    Ok(Real::from_real(self.z3_ctx, i_val, 1))
                }
                _ => Err(Z3Error::UnsupportedConstant {
                    value: value.clone(),
                    reason: "Expected a float or int constant for Z3 Real conversion.".to_string(),
                }),
            },
            Expr::Name(ExprName { id, .. }) => self.get_or_create_real_var(id.as_str()),
            _ => Err(Z3Error::UnsupportedExpressionType {
                expr_repr: format!("Cannot convert to Z3 Real: {:?}", expr), // Use expr
            }),
        }
    }

    /// Performs a single comparison between two Z3 dynamic values.
    fn perform_single_z3_comparison(
        &self,
        z3_lhs_dynamic: &Dynamic<'cfg>,
        z3_rhs_dynamic: &Dynamic<'cfg>,
        op: CmpOp,
    ) -> Result<Bool<'cfg>, Z3Error> {
        match (
            z3_lhs_dynamic.get_sort().kind(),
            z3_rhs_dynamic.get_sort().kind(),
        ) {
            (z3::SortKind::Int, z3::SortKind::Int) => {
                let l = z3_lhs_dynamic.as_int().unwrap();
                let r = z3_rhs_dynamic.as_int().unwrap();
                match op {
                    CmpOp::Eq => Ok(l._eq(&r)),
                    CmpOp::NotEq => Ok(l._eq(&r).not()),
                    CmpOp::Lt => Ok(l.lt(&r)),
                    CmpOp::LtE => Ok(l.le(&r)),
                    CmpOp::Gt => Ok(l.gt(&r)),
                    CmpOp::GtE => Ok(l.ge(&r)),
                    _ => Err(Z3Error::UnsupportedCmpOperatorForSort {
                        op,
                        sort_name: "Int".to_string(),
                    }),
                }
            }
            (z3::SortKind::Bool, z3::SortKind::Bool) => {
                let l = z3_lhs_dynamic.as_bool().unwrap();
                let r = z3_rhs_dynamic.as_bool().unwrap();
                match op {
                    CmpOp::Eq => Ok(l._eq(&r)),
                    CmpOp::NotEq => Ok(l._eq(&r).not()),
                    _ => Err(Z3Error::UnsupportedCmpOperatorForSort {
                        op,
                        sort_name: "Bool".to_string(),
                    }),
                }
            }
            (z3::SortKind::Seq, z3::SortKind::Seq) => {
                // String comparison
                let l = z3_lhs_dynamic.as_string().unwrap();
                let r = z3_rhs_dynamic.as_string().unwrap();
                match op {
                    CmpOp::Eq => Ok(l._eq(&r)),
                    CmpOp::NotEq => Ok(l._eq(&r).not()),
                    _ => Err(Z3Error::UnsupportedCmpOperatorForSort {
                        op,
                        sort_name: "String".to_string(),
                    }),
                }
            }
            (z3::SortKind::Real, z3::SortKind::Real) => {
                let l = z3_lhs_dynamic.as_real().unwrap();
                let r = z3_rhs_dynamic.as_real().unwrap();
                match op {
                    CmpOp::Eq => Ok(l._eq(&r)),
                    CmpOp::NotEq => Ok(l._eq(&r).not()),
                    CmpOp::Lt => Ok(l.lt(&r)),
                    CmpOp::LtE => Ok(l.le(&r)),
                    CmpOp::Gt => Ok(l.gt(&r)),
                    CmpOp::GtE => Ok(l.ge(&r)),
                    _ => Err(Z3Error::UnsupportedCmpOperatorForSort {
                        op,
                        sort_name: "Real".to_string(),
                    }),
                }
            }
            (z3::SortKind::Int, z3::SortKind::Real) => {
                let l_int = z3_lhs_dynamic.as_int().unwrap();
                let l = Real::from_int(&l_int);
                let r = z3_rhs_dynamic.as_real().unwrap();
                match op {
                    CmpOp::Eq => Ok(l._eq(&r)),
                    CmpOp::NotEq => Ok(l._eq(&r).not()),
                    CmpOp::Lt => Ok(l.lt(&r)),
                    CmpOp::LtE => Ok(l.le(&r)),
                    CmpOp::Gt => Ok(l.gt(&r)),
                    CmpOp::GtE => Ok(l.ge(&r)),
                    _ => Err(Z3Error::UnsupportedCmpOperatorForSort {
                        op,
                        sort_name: "Int/Real".to_string(),
                    }),
                }
            }
            (z3::SortKind::Real, z3::SortKind::Int) => {
                let l = z3_lhs_dynamic.as_real().unwrap();
                let r_int = z3_rhs_dynamic.as_int().unwrap();
                let r = Real::from_int(&r_int);
                match op {
                    CmpOp::Eq => Ok(l._eq(&r)),
                    CmpOp::NotEq => Ok(l._eq(&r).not()),
                    CmpOp::Lt => Ok(l.lt(&r)),
                    CmpOp::LtE => Ok(l.le(&r)),
                    CmpOp::Gt => Ok(l.gt(&r)),
                    CmpOp::GtE => Ok(l.ge(&r)),
                    _ => Err(Z3Error::UnsupportedCmpOperatorForSort {
                        op,
                        sort_name: "Real/Int".to_string(),
                    }),
                }
            }
            (left_kind, right_kind) => Err(Z3Error::TypeMismatchInComparison {
                op,
                left_type: format!("{:?}", left_kind),
                right_type: format!("{:?}", right_kind),
            }),
        }
    }

    /// Converts a Python AST `Expr` to a Z3 `Bool` AST.
    fn python_expr_to_z3_bool(&mut self, expr: &Expr) -> Result<Bool<'cfg>, Z3Error> {
        match &expr {
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
                let z3_operands: Result<Vec<Bool<'cfg>>, Z3Error> = values
                    .iter()
                    .map(|val_expr| self.python_expr_to_z3_bool(val_expr))
                    .collect();
                let z3_operands = z3_operands?;

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
                if ops.is_empty() || ops.len() != comparators.len() {
                    // This case should ideally not be hit if the AST is well-formed from rustpython-parser.
                    return Err(Z3Error::MalformedExpression(
                        "Compare node has mismatched ops and comparators or empty ops.".to_string(),
                    ));
                }

                let mut all_sub_comparisons: Vec<Bool<'cfg>> = Vec::with_capacity(ops.len());

                let mut current_lhs_ast_expr = left.as_ref(); // Initial LHS is the `left` field of ExprCompare

                for i in 0..ops.len() {
                    let current_rhs_ast_expr = &comparators[i];
                    let current_op = ops[i];

                    let z3_lhs_dynamic = self.python_expr_to_z3_dynamic(current_lhs_ast_expr)?;
                    let z3_rhs_dynamic = self.python_expr_to_z3_dynamic(current_rhs_ast_expr)?;

                    let sub_comparison_result = self.perform_single_z3_comparison(
                        &z3_lhs_dynamic,
                        &z3_rhs_dynamic,
                        current_op,
                    )?;
                    all_sub_comparisons.push(sub_comparison_result);

                    // For the next iteration, the current RHS becomes the next LHS
                    current_lhs_ast_expr = current_rhs_ast_expr;
                }

                if all_sub_comparisons.len() == 1 {
                    // Ok(all_sub_comparisons.remove(0)) // remove(0) is inefficient for Vec
                    Ok(all_sub_comparisons.pop().unwrap()) // More efficient for single element
                } else {
                    // For multiple comparisons (chained), AND them together.
                    let refs_to_sub_comparisons: Vec<&Bool<'_>> =
                        all_sub_comparisons.iter().collect();
                    Ok(Bool::and(self.z3_ctx, &refs_to_sub_comparisons))
                }
            }
            _ => Err(Z3Error::UnsupportedExpressionType {
                expr_repr: format!("{:?}", expr),
            }),
        }
    }

    /// Converts a Python AST `Expr` to a Z3 `Dynamic` AST.
    fn python_expr_to_z3_dynamic(&mut self, expr: &Expr) -> Result<Dynamic<'cfg>, Z3Error> {
        // Try converting to specific types first, then fallback or handle BinOp
        if let Ok(r_val) = self.python_expr_to_z3_real(expr) {
            // Try Real first
            return Ok(Dynamic::from_ast(&r_val));
        }
        if let Ok(s_val) = self.python_expr_to_z3_string(expr) {
            return Ok(Dynamic::from_ast(&s_val));
        }
        // python_expr_to_z3_bool handles UnaryOp::Not, BoolOp, and Compare internally
        if let Ok(b_val) = self.python_expr_to_z3_bool(expr) {
            return Ok(Dynamic::from_ast(&b_val));
        }
        if let Ok(i_val) = self.python_expr_to_z3_int(expr) {
            // Int last, as it could be promoted to Real
            return Ok(Dynamic::from_ast(&i_val));
        }

        // Handle UnaryOp if not caught by python_expr_to_z3_bool (e.g., for USub, UAdd)
        if let Expr::UnaryOp(ExprUnaryOp { op, operand, .. }) = expr {
            match op {
                UnaryOp::USub => {
                    let z3_operand_dynamic = self.python_expr_to_z3_dynamic(operand)?;
                    match z3_operand_dynamic.get_sort().kind() {
                        z3::SortKind::Int => {
                            let int_val = z3_operand_dynamic.as_int().ok_or_else(|| {
                                Z3Error::InternalError(
                                    "USub: Failed to cast to Int after sort check".to_string(),
                                )
                            })?;
                            return Ok(Dynamic::from_ast(&int_val.unary_minus()));
                        }
                        z3::SortKind::Real => {
                            let real_val = z3_operand_dynamic.as_real().ok_or_else(|| {
                                Z3Error::InternalError(
                                    "USub: Failed to cast to Real after sort check".to_string(),
                                )
                            })?;
                            return Ok(Dynamic::from_ast(&real_val.unary_minus()));
                        }
                        other_sort => {
                            return Err(Z3Error::UnsupportedUnaryOperatorForSort {
                                op: *op,
                                sort_name: format!("{:?}", other_sort),
                            });
                        }
                    }
                }
                UnaryOp::UAdd => {
                    // Unary plus is a no-op, just return the operand's Z3 value
                    return self.python_expr_to_z3_dynamic(operand);
                }
                UnaryOp::Not => {
                    // This should have been handled by the `python_expr_to_z3_bool(expr)` call above.
                    // If execution reaches here for UnaryOp::Not, it implies an issue with
                    // the logic flow or that `python_expr_to_z3_bool` failed unexpectedly.
                    return Err(Z3Error::InternalError(
                        "UnaryOp::Not reached unexpected location in python_expr_to_z3_dynamic"
                            .to_string(),
                    ));
                }
                UnaryOp::Invert => {
                    // Bitwise not (~) is not directly supported for Z3 Int/Real in this context.
                    return Err(Z3Error::UnsupportedUnaryOperator { op: *op });
                }
            }
        }

        if let Expr::BinOp(ExprBinOp {
            left, op, right, ..
        }) = &expr
        {
            // Use &expr
            // For BinOp, determine if it's Real or Int arithmetic
            // This is a simplification: assumes homogeneous operations or promotes Int to Real.
            // A more robust solution would inspect types more deeply or rely on type inference.
            let left_dyn = self.python_expr_to_z3_dynamic(left)?;
            let right_dyn = self.python_expr_to_z3_dynamic(right)?;

            match (left_dyn.get_sort().kind(), right_dyn.get_sort().kind()) {
                (z3::SortKind::Real, z3::SortKind::Real)
                | (z3::SortKind::Real, z3::SortKind::Int)
                | (z3::SortKind::Int, z3::SortKind::Real) => {
                    let left_val = left_dyn
                        .as_real()
                        .or_else(|| left_dyn.as_int().map(|i| Real::from_int(&i)))
                        .unwrap(); // Assuming previous logic correctly ensures these are convertible
                    let right_val = right_dyn
                        .as_real()
                        .or_else(|| right_dyn.as_int().map(|i| Real::from_int(&i)))
                        .unwrap(); // Assuming previous logic correctly ensures these are convertible
                    match op {
                        Operator::Add => Ok(Dynamic::from_ast(&Real::add(
                            self.z3_ctx,
                            &[&left_val, &right_val],
                        ))),
                        Operator::Sub => Ok(Dynamic::from_ast(&Real::sub(
                            self.z3_ctx,
                            &[&left_val, &right_val],
                        ))),
                        Operator::Mult => Ok(Dynamic::from_ast(&Real::mul(
                            self.z3_ctx,
                            &[&left_val, &right_val],
                        ))),
                        Operator::Div => Ok(Dynamic::from_ast(&Real::div(&left_val, &right_val))),
                        Operator::Pow => Ok(Dynamic::from_ast(&left_val.power(&right_val))),
                        _ => Err(Z3Error::UnsupportedExpressionType {
                            expr_repr: format!("Unsupported binary operator {:?} for Reals", op),
                        }),
                    }
                }
                (z3::SortKind::Int, z3::SortKind::Int) => {
                    let left_val = left_dyn.as_int().unwrap(); // Assuming previous logic
                    let right_val = right_dyn.as_int().unwrap(); // Assuming previous logic
                    match op {
                        Operator::Add => Ok(Dynamic::from_ast(&Int::add(
                            self.z3_ctx,
                            &[&left_val, &right_val],
                        ))),
                        Operator::Sub => Ok(Dynamic::from_ast(&Int::sub(
                            self.z3_ctx,
                            &[&left_val, &right_val],
                        ))),
                        Operator::Mult => Ok(Dynamic::from_ast(&Int::mul(
                            self.z3_ctx,
                            &[&left_val, &right_val],
                        ))),
                        Operator::Pow => Ok(Dynamic::from_ast(&left_val.power(&right_val))),
                        _ => Err(Z3Error::UnsupportedExpressionType {
                            expr_repr: format!("Unsupported binary operator {:?} for Ints", op),
                        }),
                    }
                }
                _ => Err(Z3Error::UnsupportedExpressionType {
                    expr_repr: format!(
                        "Unsupported binary operation between sorts {:?} and {:?}",
                        left_dyn.get_sort().kind(),
                        right_dyn.get_sort().kind()
                    ),
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
        match &stmt {
            // Changed to match &stmt
            Stmt::Assign(StmtAssign { targets, value, .. }) => {
                if targets.len() == 1 {
                    if let Expr::Name(ExprName { id, ctx, .. }) = &targets[0] {
                        // Use .node
                        if matches!(ctx, ExprContext::Store) {
                            let target_name = id.to_string();
                            let z3_rhs_value = self.python_expr_to_z3_dynamic(value)?;

                            // Create a new SSA variable for the assignment
                            let new_lhs_z3_var_name_prefix = format!("{}_assigned", target_name);
                            let new_lhs_z3_var = match z3_rhs_value.get_sort().kind() {
                                z3::SortKind::Bool => Dynamic::from_ast(&Bool::fresh_const(
                                    self.z3_ctx,
                                    &new_lhs_z3_var_name_prefix,
                                )),
                                z3::SortKind::Int => Dynamic::from_ast(&Int::fresh_const(
                                    self.z3_ctx,
                                    &new_lhs_z3_var_name_prefix,
                                )),
                                z3::SortKind::Seq => Dynamic::from_ast(&Z3String::fresh_const(
                                    self.z3_ctx,
                                    &new_lhs_z3_var_name_prefix,
                                )),
                                z3::SortKind::Real => Dynamic::from_ast(&Real::fresh_const(
                                    self.z3_ctx,
                                    &new_lhs_z3_var_name_prefix,
                                )), // Added Real
                                sort_kind => {
                                    return Err(Z3Error::TypeConversion(format!(
                                    "Unsupported Z3 sort {:?} for assignment target's new value",
                                    sort_kind
                                )))
                                }
                            };

                            assertions.push(new_lhs_z3_var._eq(&z3_rhs_value));
                            // Update the map to point the Python variable name to this new SSA Z3 variable
                            self.variable_map.insert(target_name, new_lhs_z3_var);
                        }
                    }
                }
            }
            Stmt::AugAssign(StmtAugAssign {
                target, op, value, ..
            }) => {
                if let Expr::Name(ExprName { id, ctx, .. }) = target.as_ref() {
                    if matches!(ctx, ExprContext::Store) {
                        let target_name = id.to_string();
                        // Current value of the target (LHS)
                        let lhs_current_z3_val = self.python_expr_to_z3_dynamic(target)?;
                        // Value of the RHS of AugAssign
                        let rhs_z3_val = self.python_expr_to_z3_dynamic(value)?;

                        let result_val_dynamic = match (
                            lhs_current_z3_val.get_sort().kind(),
                            rhs_z3_val.get_sort().kind(),
                        ) {
                            (z3::SortKind::Real, z3::SortKind::Real)
                            | (z3::SortKind::Real, z3::SortKind::Int)
                            | (z3::SortKind::Int, z3::SortKind::Real) => {
                                let lhs_real = lhs_current_z3_val
                                    .as_real()
                                    .or_else(|| {
                                        lhs_current_z3_val.as_int().map(|i| Real::from_int(&i))
                                    })
                                    .unwrap();
                                let rhs_real = rhs_z3_val
                                    .as_real()
                                    .or_else(|| rhs_z3_val.as_int().map(|i| Real::from_int(&i)))
                                    .unwrap();
                                match op {
                                    Operator::Add => Dynamic::from_ast(&Real::add(
                                        self.z3_ctx,
                                        &[&lhs_real, &rhs_real],
                                    )),
                                    Operator::Sub => Dynamic::from_ast(&Real::sub(
                                        self.z3_ctx,
                                        &[&lhs_real, &rhs_real],
                                    )),
                                    Operator::Mult => Dynamic::from_ast(&Real::mul(
                                        self.z3_ctx,
                                        &[&lhs_real, &rhs_real],
                                    )),
                                    Operator::Div => {
                                        Dynamic::from_ast(&Real::div(&lhs_real, &rhs_real))
                                    }
                                    Operator::Pow => {
                                        // Added Power support for Reals
                                        Dynamic::from_ast(&lhs_real.power(&rhs_real))
                                    }
                                    _ => {
                                        return Err(Z3Error::UnsupportedExpressionType {
                                            expr_repr: format!(
                                                "Unsupported AugAssign operator {:?} for Reals",
                                                op
                                            ),
                                        })
                                    }
                                }
                            }
                            (z3::SortKind::Int, z3::SortKind::Int) => {
                                let lhs_int = lhs_current_z3_val.as_int().unwrap();
                                let rhs_int = rhs_z3_val.as_int().unwrap();
                                match op {
                                    Operator::Add => Dynamic::from_ast(&Int::add(
                                        self.z3_ctx,
                                        &[&lhs_int, &rhs_int],
                                    )),
                                    Operator::Sub => Dynamic::from_ast(&Int::sub(
                                        self.z3_ctx,
                                        &[&lhs_int, &rhs_int],
                                    )),
                                    Operator::Mult => Dynamic::from_ast(&Int::mul(
                                        self.z3_ctx,
                                        &[&lhs_int, &rhs_int],
                                    )),
                                    Operator::Pow => {
                                        // Added Power support for Ints
                                        Dynamic::from_ast(&lhs_int.power(&rhs_int))
                                    }
                                    // Note: Operator::Div for Ints would typically promote to Real in Python for /=
                                    // If Operator::FloorDiv (//=) was intended, Int::div would be used.
                                    // Current structure correctly promotes to Real if Operator::Div is used.
                                    _ => {
                                        return Err(Z3Error::UnsupportedExpressionType {
                                            expr_repr: format!(
                                                "Unsupported AugAssign operator {:?} for Ints",
                                                op
                                            ),
                                        })
                                    }
                                }
                            }
                            (lk, rk) => {
                                return Err(Z3Error::TypeConversion(format!(
                                    "Unsupported types for AugAssign: {:?} and {:?}",
                                    lk, rk
                                )))
                            }
                        };

                        let new_lhs_z3_var_name_prefix = format!("{}_aug_assigned", target_name);
                        let new_lhs_z3_var = match result_val_dynamic.get_sort().kind() {
                            z3::SortKind::Int => Dynamic::from_ast(&Int::fresh_const(
                                self.z3_ctx,
                                &new_lhs_z3_var_name_prefix,
                            )),
                            z3::SortKind::Real => Dynamic::from_ast(&Real::fresh_const(
                                self.z3_ctx,
                                &new_lhs_z3_var_name_prefix,
                            )),
                            // Assuming Bool and String aug assigns are not standard or handled elsewhere if needed
                            sort_kind => {
                                return Err(Z3Error::TypeConversion(format!(
                                "Unsupported Z3 sort {:?} for aug-assignment target's new value",
                                sort_kind
                            )))
                            }
                        };

                        assertions.push(new_lhs_z3_var._eq(&result_val_dynamic));
                        self.variable_map.insert(target_name, new_lhs_z3_var);
                    }
                }
            }
            Stmt::Expr(StmtExpr { value, .. }) => {
                // Handle 'assert' calls if they appear as StmtExpr
                if let Expr::Call(call_expr) = &**value {
                    if let Expr::Name(name_expr) = &*call_expr.func {
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
            Stmt::Pass(_) => {} // No Z3 constraints for pass
            _ => {}             // Other statement types not directly contributing constraints here
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
            let stmts_to_process = match node {
                Node::Cond { stmts, .. }
                | Node::Return { stmts, .. }
                | Node::Raise { stmts, .. } => stmts,
            };
            for stmt in stmts_to_process {
                self.process_statement_for_z3(stmt, &mut path_assertions)?;
            }

            if let Node::Cond {
                expr: condition_expr,
                ..
            } = node
            {
                let condition_ast = self.python_expr_to_z3_bool(condition_expr)?;
                match edge {
                    Edge::True => path_assertions.push(condition_ast),
                    Edge::False => path_assertions.push(condition_ast.not()),
                    Edge::Terminal => {}
                }
            }
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

pub fn analyze_paths(
    all_paths: &[Vec<(NodeId, Edge)>],
    cfg_data: &ControlFlowGraph,
) -> Vec<PathConstraintResult> {
    let z3_config = Config::new();
    let z3_ctx = Context::new(&z3_config);
    let mut constraint_generator =
        match Z3ConstraintGenerator::new(&z3_ctx, cfg_data.get_arguments()) {
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

    all_paths
        .iter()
        .enumerate()
        .map(|(i, path)| {
            if let Err(reinit_err) =
                constraint_generator.clear_and_reinitialize_args(cfg_data.get_arguments())
            {
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
            path_result
        })
        .collect()
}

#[allow(dead_code)]
pub fn print_paths_analysis(cfg: &ControlFlowGraph, paths: &[Vec<(NodeId, Edge)>]) {
    let constraint_results = analyze_paths(paths, cfg);
    println!("=== PATH ANALYSIS RESULTS ===");
    println!("Total paths analyzed: {}\n", constraint_results.len());
    for result in constraint_results {
        println!(
            "Path {}: Satisfiable = {}",
            result.path_index, result.is_satisfiable
        );
        if let Some(path_structure) = paths.get(result.path_index) {
            println!("  Path Structure: {:?}", path_structure);
        }
        if let Some(model_str) = &result.model {
            println!("  Z3 Model:");
            model_str
                .lines()
                .filter(|line| !line.trim().is_empty())
                .for_each(|line| {
                    println!("    {}", line.trim());
                });
        }
        if let Some(err) = &result.error {
            println!("  Error: {}", err);
        }
        println!();
    }
}
