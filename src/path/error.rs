use crate::cfg::NodeId;
use rustpython_ast::{BoolOp, CmpOp, Constant, UnaryOp}; // Using user's import style
use thiserror::Error;

// Error type for path analysis operations.
#[derive(Error, Debug, Clone)]
pub enum PathError {
    /// Error when the control flow graph is empty.
    #[error("Control flow graph is empty, cannot extract paths.")]
    EmptyGraph,

    /// Error when no paths are found in the control flow graph.
    #[error("No paths found in the control flow graph.")]
    NoPathsFound,
}

// Error type for Z3 operations and AST to Z3 conversion.
#[derive(Error, Debug, Clone)]
pub enum Z3Error {
    /// Error when initializing the Z3 context.
    #[error("Failed to initialize Z3 context: {0}")]
    Initialization(String),

    /// Error when creating a Z3 solver.
    #[error("Failed to create Z3 solver: {0}")]
    SolverCreation(String),

    /// Error when Z3 solver returns an unknown status.
    #[error("Z3 solver returned unknown: {0}")]
    SolverUnknown(String),

    /// General error during Z3 type conversion.
    #[error("Z3 type conversion error: {0}")]
    TypeConversion(String),

    /// Error when a Python constant cannot be converted to the expected Z3 type.
    #[error("Unsupported constant for Z3 conversion: {value:?}, reason: {reason}")]
    UnsupportedConstant { value: Constant, reason: String },

    /// Error when a Python expression type is not supported for Z3 conversion.
    #[error("Unsupported expression type for Z3 conversion: {expr_repr}")]
    UnsupportedExpressionType { expr_repr: String },

    /// Error when a Python unary operator is not supported.
    #[error("Unsupported unary operator for Z3 conversion: {op:?}")]
    UnsupportedUnaryOperator { op: UnaryOp },

    /// Error when a Python boolean operator is not supported.
    #[error("Unsupported boolean operator for Z3 conversion: {op:?}")]
    UnsupportedBoolOperator { op: BoolOp },

    /// Error when a Python comparison operator is not supported.
    #[error("Unsupported comparison operator for Z3 conversion: {op:?}")]
    UnsupportedCmpOperator { op: CmpOp },

    /// Error when a boolean operation (e.g., And, Or) has no values.
    #[error("Boolean operation received empty values list")]
    EmptyBoolOpValues,

    /// Error for chained comparisons (e.g., a < b < c) which are not directly supported.
    #[error(
        "Chained comparisons or multiple operators in a single Compare node are not supported"
    )]
    ChainedComparisonNotSupported,

    /// Error when there is a type mismatch for a variable in Z3 operations.
    #[error("Type mismatch for variable '{variable_name}': expected {expected_type}, found other")]
    TypeMismatch {
        variable_name: String,
        expected_type: String,
    },

    /// Error when a NodeId is not found in the ControlFlowGraph.
    #[error("NodeId {0} not found in CFG")]
    NodeNotFoundInCfg(NodeId),

    /// Error related to specific expression variants not being handled.
    #[error("Unhandled expression variant: {variant_name:?}")]
    UnhandledExpressionVariant { variant_name: String },
}
