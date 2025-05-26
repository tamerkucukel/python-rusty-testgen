mod ast_loader;
mod cfg;
mod path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python_file_path = "./test-file.py";
    let python_ast = ast_loader::load_ast_from_file(python_file_path)?;
    for func_def in python_ast {
        let mut cfg = cfg::ControlFlowGraph::new();
        cfg.from_ast(func_def);

        if let Some(paths) = path::Explorer::get_paths(&cfg) {
            println!("Paths in the control flow graph: {:?}", paths);
        }

        for (node_id, node) in cfg.get_graph() {
            println!("Node ID: {:?}, Node: {:#?}\n", node_id, node);
        }
    }

    Ok(())
}
