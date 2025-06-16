use crate::cfg::{ControlFlowGraph, Edge, Node, NodeId};
use rustpython_ast::{
    BoolOp, CmpOp, Constant, Expr, ExprBinOp, ExprBoolOp, ExprCompare, ExprConstant, ExprContext,
    ExprName, ExprUnaryOp, Operator, Stmt, StmtAssert, StmtAssign, StmtAugAssign, StmtExpr,
    UnaryOp,
};
use std::collections::HashMap;
use z3::ast::{Ast, Bool, Dynamic, Int, Real, String as Z3String};
use z3::{Config, Context, SatResult, Solver};

use super::error::Z3Error;

/// Stores the result of a Z3 satisfiability check for a single path through a function.
///
/// Each `PathConstraintResult` corresponds to one of the possible execution paths
/// identified by control flow graph analysis.
#[derive(Debug, Clone)]
pub struct PathConstraintResult {
    /// Index of the path in the original list of all paths provided to `analyze_paths`.
    pub path_index: usize,
    /// Indicates whether the constraints for this path were satisfiable.
    pub is_satisfiable: bool,
    /// A string representation of the Z3 model (i.e., variable assignments)
    /// if the path is satisfiable. `None` otherwise.
    pub model: Option<String>,
    /// Any error encountered during constraint generation or solving for this specific path.
    /// `None` if no error occurred.
    pub error: Option<Z3Error>,
}

/// Generates Z3 constraints from Python expressions and statements along a given execution path.
///
/// This struct is responsible for translating Python AST elements into their logical
/// equivalents in Z3. It maintains a `variable_map` to track Python variables
/// and their corresponding Z3 AST nodes. To handle variable reassignments, it employs
/// an SSA-like approach by creating new Z3 variables for each assignment.
struct Z3ConstraintGenerator<'cfg> {
    z3_ctx: &'cfg Context,
    /// Maps Python variable names (strings) to their current Z3 `Dynamic` AST node
    /// representations within the current path being analyzed.
    variable_map: HashMap<String, Dynamic<'cfg>>,
}

/// Macro to simplify the common pattern of retrieving an existing Z3 typed variable
/// from `variable_map` or creating and inserting it if not found.
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
    ///
    /// Initializes the `variable_map` with Z3 variables representing the function's arguments,
    /// typed according to the provided `type_hint_opt`.
    fn new(
        ctx: &'cfg Context,
        function_args: &[(String, Option<String>)],
    ) -> Result<Self, Z3Error> {
        let mut variable_map = HashMap::new();
        for (name, type_hint_opt) in function_args {
            if let Some(type_hint) = type_hint_opt {
                let z3_var_dynamic = match type_hint.as_str() {
                    "int" => Dynamic::from_ast(&Int::new_const(ctx, name.as_str())),
                    "bool" => Dynamic::from_ast(&Bool::new_const(ctx, name.as_str())),
                    "str" => Dynamic::from_ast(&Z3String::new_const(ctx, name.as_str())),
                    "float" => Dynamic::from_ast(&Real::new_const(ctx, name.as_str())),
                    _ => continue, // Unknown type hints are ignored for initial var creation
                };
                variable_map.insert(name.clone(), z3_var_dynamic);
            }
        }
        Ok(Z3ConstraintGenerator {
            z3_ctx: ctx,
            variable_map,
        })
    }

    /// Clears the current `variable_map` and re-initializes it with Z3 variables
    /// for the function arguments.
    ///
    /// This method is crucial for ensuring that constraint generation for each new
    /// path starts with a fresh state for function arguments, preventing interference
    /// between path analyses.
    fn clear_and_reinitialize_args(
        &mut self,
        function_args: &[(String, Option<String>)],
    ) -> Result<(), Z3Error> {
        self.variable_map.clear();
        // Re-populate based on function arguments, similar to `new`
        for (name, type_hint_opt) in function_args {
            if let Some(type_hint) = type_hint_opt {
                let z3_var_dynamic = match type_hint.as_str() {
                    "int" => Dynamic::from_ast(&Int::new_const(self.z3_ctx, name.as_str())),
                    "bool" => Dynamic::from_ast(&Bool::new_const(self.z3_ctx, name.as_str())),
                    "str" => Dynamic::from_ast(&Z3String::new_const(self.z3_ctx, name.as_str())),
                    "float" => Dynamic::from_ast(&Real::new_const(self.z3_ctx, name.as_str())),
                    _ => continue,
                };
                self.variable_map.insert(name.clone(), z3_var_dynamic);
            }
        }
        Ok(())
    }

    /// Retrieves an existing Z3 integer variable by `name` or creates a new one.
    fn get_or_create_int_var(&mut self, name: &str) -> Result<Int<'cfg>, Z3Error> {
        get_or_create_z3_var!(self, name, Int<'cfg>, Int::new_const, as_int, "Int")
    }

    /// Retrieves an existing Z3 boolean variable by `name` or creates a new one.
    fn get_or_create_bool_var(&mut self, name: &str) -> Result<Bool<'cfg>, Z3Error> {
        get_or_create_z3_var!(self, name, Bool<'cfg>, Bool::new_const, as_bool, "Bool")
    }

    /// Retrieves an existing Z3 string variable by `name` or creates a new one.
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

    /// Retrieves an existing Z3 real variable by `name` or creates a new one.
    fn get_or_create_real_var(&mut self, name: &str) -> Result<Real<'cfg>, Z3Error> {
        get_or_create_z3_var!(self, name, Real<'cfg>, Real::new_const, as_real, "Real")
    }

    /// Converts a Python AST `Expr::Constant` (integer) to a Z3 `Int` AST node.
    fn python_expr_to_z3_int(&mut self, expr: &Expr) -> Result<Int<'cfg>, Z3Error> {
        match expr {
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
                expr_repr: format!("Cannot convert to Z3 Int: {:?}", expr),
            }),
        }
    }

    /// Converts a Python AST `Expr::Constant` (string) to a Z3 `String` AST node.
    fn python_expr_to_z3_string(&mut self, expr: &Expr) -> Result<Z3String<'cfg>, Z3Error> {
        match expr {
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
                expr_repr: format!("Cannot convert to Z3 String: {:?}", expr),
            }),
        }
    }

    /// Converts a Python AST `Expr::Constant` (float or int) to a Z3 `Real` AST node.
    fn python_expr_to_z3_real(&mut self, expr: &Expr) -> Result<Real<'cfg>, Z3Error> {
        match expr {
            Expr::Constant(ExprConstant { value, .. }) => match value {
                Constant::Float(f) => {
                    // NOTE: The following conversion of a Python float to a Z3 Real
                    // truncates the float to an integer (f.to_string().parse::<i32>().unwrap_or(0)).
                    // This is preserved to adhere to the "no logic change" requirement.
                    // A more accurate conversion would involve parsing the float string
                    // or converting to a rational representation if logic changes were permitted.
                    Ok(Real::from_real(
                        self.z3_ctx,
                        f.to_string().parse::<i32>().unwrap_or(0), // Original logic
                        1,
                    ))
                }
                Constant::Int(i) => {
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
                expr_repr: format!("Cannot convert to Z3 Real: {:?}", expr),
            }),
        }
    }

    /// Performs a single comparison (e.g., <, ==, >=) between two Z3 `Dynamic` values.
    ///
    /// Handles type promotions, such as comparing an `Int` with a `Real`.
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
                let l = z3_lhs_dynamic.as_int().ok_or_else(|| {
                    Z3Error::InternalError("LHS expected to be Int after sort check".to_string())
                })?;
                let r = z3_rhs_dynamic.as_int().ok_or_else(|| {
                    Z3Error::InternalError("RHS expected to be Int after sort check".to_string())
                })?;
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
                let l = z3_lhs_dynamic.as_bool().ok_or_else(|| {
                    Z3Error::InternalError("LHS expected to be Bool after sort check".to_string())
                })?;
                let r = z3_rhs_dynamic.as_bool().ok_or_else(|| {
                    Z3Error::InternalError("RHS expected to be Bool after sort check".to_string())
                })?;
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
                let l = z3_lhs_dynamic.as_string().ok_or_else(|| {
                    Z3Error::InternalError("LHS expected to be String after sort check".to_string())
                })?;
                let r = z3_rhs_dynamic.as_string().ok_or_else(|| {
                    Z3Error::InternalError("RHS expected to be String after sort check".to_string())
                })?;
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
                let l = z3_lhs_dynamic.as_real().ok_or_else(|| {
                    Z3Error::InternalError("LHS expected to be Real after sort check".to_string())
                })?;
                let r = z3_rhs_dynamic.as_real().ok_or_else(|| {
                    Z3Error::InternalError("RHS expected to be Real after sort check".to_string())
                })?;
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
                let l_int = z3_lhs_dynamic.as_int().ok_or_else(|| {
                    Z3Error::InternalError(
                        "LHS expected to be Int for Int/Real comparison".to_string(),
                    )
                })?;
                let l = Real::from_int(&l_int);
                let r = z3_rhs_dynamic.as_real().ok_or_else(|| {
                    Z3Error::InternalError(
                        "RHS expected to be Real for Int/Real comparison".to_string(),
                    )
                })?;
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
                let l = z3_lhs_dynamic.as_real().ok_or_else(|| {
                    Z3Error::InternalError(
                        "LHS expected to be Real for Real/Int comparison".to_string(),
                    )
                })?;
                let r_int = z3_rhs_dynamic.as_int().ok_or_else(|| {
                    Z3Error::InternalError(
                        "RHS expected to be Int for Real/Int comparison".to_string(),
                    )
                })?;
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

    /// Converts a Python AST `Expr` to a Z3 `Bool` AST node.
    ///
    /// This handles boolean constants, variable names, `not` operations,
    /// `and`/`or` operations, and comparison operations (e.g., `x < y`, `a == b`).
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
                let z3_operands: Vec<Bool<'cfg>> = values
                    .iter()
                    .map(|val_expr| self.python_expr_to_z3_bool(val_expr))
                    .collect::<Result<_, _>>()?;

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
                    return Err(Z3Error::MalformedExpression(
                        "Compare node has mismatched ops and comparators or empty ops.".to_string(),
                    ));
                }

                let mut all_sub_comparisons: Vec<Bool<'cfg>> = Vec::with_capacity(ops.len());
                let mut current_lhs_ast_expr = left.as_ref();

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

                    current_lhs_ast_expr = current_rhs_ast_expr; // Chain comparisons
                }

                if all_sub_comparisons.len() == 1 {
                    Ok(all_sub_comparisons
                        .pop()
                        .expect("Vector should have one element"))
                } else {
                    let refs_to_sub_comparisons: Vec<&Bool<'_>> =
                        all_sub_comparisons.iter().collect();
                    Ok(Bool::and(self.z3_ctx, &refs_to_sub_comparisons))
                }
            }
            _ => Err(Z3Error::UnsupportedExpressionType {
                expr_repr: format!("Cannot convert to Z3 Bool: {:?}", expr),
            }),
        }
    }

    /// Converts a Python AST `Expr` to a Z3 `Dynamic` AST node.
    ///
    /// This method acts as a dispatcher, attempting to convert the expression
    /// to the most specific Z3 type possible (Real, String, Bool, Int) before
    /// handling more complex structures like `UnaryOp` (for arithmetic negation)
    /// or `BinOp`. The order of attempts is significant for type preference.
    fn python_expr_to_z3_dynamic(&mut self, expr: &Expr) -> Result<Dynamic<'cfg>, Z3Error> {
        // Attempt conversion to specific Z3 types with defined precedence.
        // Real is tried first, then String, then Bool (which handles comparisons), then Int.
        if let Ok(r_val) = self.python_expr_to_z3_real(expr) {
            return Ok(Dynamic::from_ast(&r_val));
        }
        if let Ok(s_val) = self.python_expr_to_z3_string(expr) {
            return Ok(Dynamic::from_ast(&s_val));
        }
        if let Ok(b_val) = self.python_expr_to_z3_bool(expr) {
            // python_expr_to_z3_bool handles UnaryOp::Not, BoolOp, and Compare internally.
            return Ok(Dynamic::from_ast(&b_val));
        }
        if let Ok(i_val) = self.python_expr_to_z3_int(expr) {
            return Ok(Dynamic::from_ast(&i_val));
        }

        // Handle UnaryOp if not already handled by python_expr_to_z3_bool (e.g., USub, UAdd).
        if let Expr::UnaryOp(ExprUnaryOp { op, operand, .. }) = expr {
            match op {
                UnaryOp::USub => {
                    let z3_operand_dynamic = self.python_expr_to_z3_dynamic(operand)?;
                    match z3_operand_dynamic.get_sort().kind() {
                        z3::SortKind::Int => {
                            let int_val = z3_operand_dynamic.as_int().ok_or_else(|| {
                                Z3Error::InternalError(
                                    "USub operand expected to be Int after sort check".to_string(),
                                )
                            })?;
                            Ok(Dynamic::from_ast(&int_val.unary_minus()))
                        }
                        z3::SortKind::Real => {
                            let real_val = z3_operand_dynamic.as_real().ok_or_else(|| {
                                Z3Error::InternalError(
                                    "USub operand expected to be Real after sort check".to_string(),
                                )
                            })?;
                            Ok(Dynamic::from_ast(&real_val.unary_minus()))
                        }
                        other_sort => Err(Z3Error::UnsupportedUnaryOperatorForSort {
                            op: *op,
                            sort_name: format!("{:?}", other_sort),
                        }),
                    }
                }
                UnaryOp::UAdd => self.python_expr_to_z3_dynamic(operand), // UAdd is a no-op.
                UnaryOp::Not => Err(Z3Error::InternalError(
                    // Should be handled by python_expr_to_z3_bool.
                    "UnaryOp::Not reached unexpected location in python_expr_to_z3_dynamic"
                        .to_string(),
                )),
                UnaryOp::Invert => Err(Z3Error::UnsupportedUnaryOperator { op: *op }), // Bitwise not (~)
            }
        }
        // Handle Binary Operations (arithmetic).
        else if let Expr::BinOp(ExprBinOp {
            left, op, right, ..
        }) = expr
        {
            let left_dyn = self.python_expr_to_z3_dynamic(left)?;
            let right_dyn = self.python_expr_to_z3_dynamic(right)?;

            match (left_dyn.get_sort().kind(), right_dyn.get_sort().kind()) {
                (z3::SortKind::Real, z3::SortKind::Real)
                | (z3::SortKind::Real, z3::SortKind::Int)
                | (z3::SortKind::Int, z3::SortKind::Real) => {
                    let left_val = left_dyn
                        .as_real()
                        .or_else(|| {
                            left_dyn
                                .as_int()
                                .map(|ast: z3::ast::Int<'_>| Real::from_int(&ast))
                        })
                        .ok_or_else(|| {
                            Z3Error::TypeConversion(
                                "Failed to convert LHS to Real for BinOp".to_string(),
                            )
                        })?;
                    let right_val = right_dyn
                        .as_real()
                        .or_else(|| {
                            right_dyn
                                .as_int()
                                .map(|ast: z3::ast::Int<'_>| Real::from_int(&ast))
                        })
                        .ok_or_else(|| {
                            Z3Error::TypeConversion(
                                "Failed to convert RHS to Real for BinOp".to_string(),
                            )
                        })?;
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
                    let left_val = left_dyn.as_int().ok_or_else(|| {
                        Z3Error::TypeConversion(
                            "Failed to convert LHS to Int for BinOp".to_string(),
                        )
                    })?;
                    let right_val = right_dyn.as_int().ok_or_else(|| {
                        Z3Error::TypeConversion(
                            "Failed to convert RHS to Int for BinOp".to_string(),
                        )
                    })?;
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
                (lk, rk) => Err(Z3Error::UnsupportedExpressionType {
                    expr_repr: format!(
                        "Unsupported binary operation between sorts {:?} and {:?}",
                        lk, rk
                    ),
                }),
            }
        }
        // If no conversion or known operation matched.
        else {
            Err(Z3Error::UnsupportedExpressionType {
                expr_repr: format!("Cannot convert to Z3 Dynamic: {:?}", expr),
            })
        }
    }

    /// Processes a single Python `Stmt` (statement) to generate corresponding Z3 assertions
    /// and update the `variable_map` for assignments.
    fn process_statement_for_z3(
        &mut self,
        stmt: &Stmt,
        assertions: &mut Vec<Bool<'cfg>>,
    ) -> Result<(), Z3Error> {
        match stmt {
            Stmt::Assign(StmtAssign { targets, value, .. }) => {
                if targets.len() == 1 {
                    // Simple assignment: x = value
                    if let Expr::Name(ExprName { id, ctx, .. }) = &targets[0] {
                        if matches!(ctx, ExprContext::Store) {
                            let target_name = id.to_string();
                            let z3_rhs_value = self.python_expr_to_z3_dynamic(value)?;

                            // Create a new SSA variable for the assignment target.
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
                                )),
                                sort_kind => {
                                    return Err(Z3Error::TypeConversion(format!(
                                    "Unsupported Z3 sort {:?} for new SSA variable in assignment",
                                    sort_kind
                                )))
                                }
                            };

                            assertions.push(new_lhs_z3_var._eq(&z3_rhs_value));
                            self.variable_map.insert(target_name, new_lhs_z3_var);
                            // Update map
                        }
                    }
                }
                // Note: Multiple assignment targets (e.g., a, b = 1, 2) are not handled here.
            }
            Stmt::AugAssign(StmtAugAssign {
                target, op, value, ..
            }) => {
                if let Expr::Name(ExprName { id, ctx, .. }) = target.as_ref() {
                    if matches!(ctx, ExprContext::Store) {
                        let target_name = id.to_string();
                        let lhs_current_z3_val = self.python_expr_to_z3_dynamic(target)?;
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
                                        lhs_current_z3_val
                                            .as_int()
                                            .map(|ast: z3::ast::Int<'_>| Real::from_int(&ast))
                                    })
                                    .ok_or_else(|| {
                                        Z3Error::TypeConversion(
                                            "Failed to convert LHS to Real for AugAssign"
                                                .to_string(),
                                        )
                                    })?;
                                let rhs_real = rhs_z3_val
                                    .as_real()
                                    .or_else(|| {
                                        rhs_z3_val
                                            .as_int()
                                            .map(|ast: z3::ast::Int<'_>| Real::from_int(&ast))
                                    })
                                    .ok_or_else(|| {
                                        Z3Error::TypeConversion(
                                            "Failed to convert RHS to Real for AugAssign"
                                                .to_string(),
                                        )
                                    })?;
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
                                    Operator::Pow => Dynamic::from_ast(&lhs_real.power(&rhs_real)),
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
                                let lhs_int = lhs_current_z3_val.as_int().ok_or_else(|| {
                                    Z3Error::TypeConversion(
                                        "Failed to convert LHS to Int for AugAssign".to_string(),
                                    )
                                })?;
                                let rhs_int = rhs_z3_val.as_int().ok_or_else(|| {
                                    Z3Error::TypeConversion(
                                        "Failed to convert RHS to Int for AugAssign".to_string(),
                                    )
                                })?;
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
                                    Operator::Pow => Dynamic::from_ast(&lhs_int.power(&rhs_int)),
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
                                    "Unsupported types for AugAssign operation: {:?} and {:?}",
                                    lk, rk
                                )))
                            }
                        };

                        // Create new SSA variable for the result of AugAssign.
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
                            sort_kind => {
                                return Err(Z3Error::TypeConversion(format!(
                                    "Unsupported Z3 sort {:?} for new SSA variable in AugAssign",
                                    sort_kind
                                )))
                            }
                        };

                        assertions.push(new_lhs_z3_var._eq(&result_val_dynamic));
                        self.variable_map.insert(target_name, new_lhs_z3_var); // Update map
                    }
                }
            }
            Stmt::Expr(StmtExpr { value, .. }) => {
                // Handle 'assert' calls if they appear as StmtExpr (e.g. `assert(x > 0)`)
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
            Stmt::Pass(_) => {} // Pass statements generate no constraints.
            // Other statement types (e.g., If, For, While, Return, Raise) are handled by
            // path traversal logic in `create_path_assertion` or are terminal.
            _ => {}
        }
        Ok(())
    }

    /// Aggregates all Z3 assertions for a given execution path into a single Z3 `Bool` AST node.
    ///
    /// It iterates through the nodes and edges of a path:
    /// - Processes statements within each node to generate assertions (e.g., from assignments).
    /// - For conditional nodes, asserts the truthiness or falsiness of the condition
    ///   based on the edge taken (True or False branch).
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

            // Process statements within the current node that contribute to path constraints
            let stmts_to_process = match node {
                Node::Cond { stmts, .. }
                | Node::Return { stmts, .. }
                | Node::Raise { stmts, .. } => stmts, // Statements executed before condition/return/raise
            };
            for stmt in stmts_to_process {
                self.process_statement_for_z3(stmt, &mut path_assertions)?;
            }

            // If the current node is a conditional, add an assertion for the branch taken
            if let Node::Cond {
                expr: condition_expr,
                ..
            } = node
            {
                let condition_ast = self.python_expr_to_z3_bool(condition_expr)?;
                match edge {
                    Edge::True => path_assertions.push(condition_ast),
                    Edge::False => path_assertions.push(condition_ast.not()),
                    Edge::Terminal => {} // Terminal edges from Cond nodes usually don't add constraints here
                }
            }
        }

        if path_assertions.is_empty() {
            // An empty path or a path with no constraining statements is trivially true.
            Ok(Bool::from_bool(self.z3_ctx, true))
        } else {
            // Combine all path assertions with AND.
            Ok(Bool::and(
                self.z3_ctx,
                &path_assertions.iter().collect::<Vec<_>>(),
            ))
        }
    }
}

/// Analyzes a list of execution paths for a function using Z3.
///
/// For each path, it generates Z3 constraints, checks for satisfiability,
/// and attempts to retrieve a model if satisfiable.
///
/// # Arguments
/// * `all_paths` - A slice of paths, where each path is a sequence of (NodeId, Edge) tuples.
/// * `cfg_data` - The Control Flow Graph data for the function being analyzed.
///
/// # Returns
/// A vector of `PathConstraintResult`, one for each input path.
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
                // If generator creation fails, all paths fail with this error.
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
        .map(|(path_idx, path)| {
            // Ensure variable map is reset for each path to reflect initial arg state.
            if let Err(reinit_err) =
                constraint_generator.clear_and_reinitialize_args(cfg_data.get_arguments())
            {
                return PathConstraintResult {
                    path_index: path_idx,
                    is_satisfiable: false,
                    model: None,
                    error: Some(reinit_err),
                };
            }

            let solver = Solver::new(&z3_ctx);
            let mut path_result = PathConstraintResult {
                path_index: path_idx,
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
                            path_result.is_satisfiable = false; // Treat Unknown as Unsat for test generation
                            path_result.error = Some(Z3Error::SolverUnknown(
                                solver
                                    .get_reason_unknown()
                                    .unwrap_or_else(|| "Reason unknown".to_string()),
                            ));
                        }
                    }
                }
                Err(e) => {
                    // Error during constraint generation for this path
                    path_result.error = Some(e);
                }
            }
            path_result
        })
        .collect()
}

/// Utility function to print the results of path analysis to the console.
///
/// This function is primarily for debugging and demonstration purposes.
#[allow(dead_code)] // Allow dead code as this is a utility/debug function
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
            println!("  Path Structure (NodeId, Edge):");
            for (node_id, edge) in path_structure {
                println!("    ({:?}, {:?})", node_id, edge);
            }
        }
        if let Some(model_str) = &result.model {
            println!("  Z3 Model:");
            model_str
                .lines()
                .filter(|line| !line.trim().is_empty()) // Filter out empty lines
                .for_each(|line| {
                    println!("    {}", line.trim()); // Indent model lines
                });
        }
        if let Some(err) = &result.error {
            println!("  Error: {}", err); // Display any errors
        }
        println!(); // Blank line for readability
    }
}
