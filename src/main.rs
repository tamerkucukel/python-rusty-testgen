mod ast_loader;
mod cfg;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python_file_path = "./test-file.py";
    let python_ast = ast_loader::load_ast_from_file(python_file_path)?;
    for func_def in python_ast {
        let mut cfg = cfg::ControlFlowGraph::new();
        cfg.from_ast(func_def);
        cfg.print_paths();
    }

    Ok(())
}
