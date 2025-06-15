# Python Rusty TestGen

Python Rusty TestGen is a command-line tool written in Rust that automatically generates `pytest` unit tests for Python functions. It analyzes Python code, explores execution paths, and uses the Z3 SMT solver to determine input values that cover these paths, then generates test cases based on this analysis.

## Overview

The primary goal of this project is to reduce the manual effort involved in writing unit tests by automatically generating test cases that cover different logical paths within Python functions. It leverages static analysis techniques and an SMT solver to achieve this.

## Features

*   **AST-based Analysis:** Parses Python code into an Abstract Syntax Tree for detailed inspection.
*   **Control Flow Graph (CFG) Generation:** Builds a CFG for each function to understand its structure and execution flows.
*   **Path Exploration:** Identifies distinct execution paths through the CFG.
*   **SMT Solver Integration (Z3):**
    *   Generates logical constraints for each path.
    *   Uses Z3 to find satisfying input models (values for function arguments) for these paths.
*   **Pytest Test Generation:**
    *   Creates `pytest` test functions based on the Z3 models.
    *   Handles various data types for arguments and return values (`int`, `bool`, `str`, `float`).
    *   Supports assertions for return values, including `pytest.approx` for floating-point comparisons.
    *   Generates tests for paths leading to `raise` statements using `pytest.raises`.
    *   Handles explicit and implicit `None` returns.
    *   Supports chained comparisons (e.g., `a < b <= c`).
*   **Logging:** Provides verbose logging (`testgen.log`) and CFG details (`cfg_details.log`).

## How it Works (High-Level Workflow)

1.  **Load AST:** The input Python file is parsed into an Abstract Syntax Tree (AST).
2.  **Build CFG:** For each function definition in the AST, a Control Flow Graph (CFG) is constructed.
3.  **Explore Paths:** The CFG is traversed to identify all unique execution paths from the function's entry to its exit points (return or raise statements).
4.  **Generate Constraints & Solve:**
    *   For each path, a set of logical constraints is generated. These constraints represent the conditions that must be true for that specific path to be taken.
    *   The Z3 SMT solver is used to find a model (a set of input values for the function's arguments) that satisfies these constraints.
5.  **Generate Tests:**
    *   If Z3 finds a satisfiable model for a path, a `pytest` test case is generated.
    *   The model values are used as inputs for the function call in the test.
    *   The expected outcome (return value or raised exception) for that path is asserted.
6.  **Output:** The generated tests are written to `test_generated_suite.py`.

## Prerequisites

*   **Rust Toolchain:** Ensure you have Rust installed (see [rustup.rs](https://rustup.rs/)).
*   **Z3 SMT Solver:** Z3 must be installed and accessible in your system's `PATH` or its library files available for linking. You can download Z3 from its [GitHub releases page](https://github.com/Z3Prover/z3/releases).
*   **Python Environment:** A Python environment with `pytest` installed is needed to run the generated tests.
    ```bash
    pip install pytest
    ```

## Building the Project

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd python-rusty-testgen
    ```
2.  Build the project:
    *   For a development build:
        ```bash
        cargo build
        ```
    *   For a release build (recommended for usage):
        ```bash
        cargo build --release
        ```
    The executable will be located at `target/debug/python-rusty-testgen` or `target/release/python-rusty-testgen`.

## Usage

Run the tool from the command line, providing the path to the Python file you want to analyze.

```bash
./target/release/python-rusty-testgen <path_to_your_python_file.py>