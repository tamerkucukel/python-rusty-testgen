use std::collections::HashMap;

use crate::cfg::{ControlFlowGraph, Edge, Node, NodeId};

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

    /// Prints all paths in the control flow graph and the graph nodes in a readable format.
    pub fn print_paths(control_flow_graph: &ControlFlowGraph) {
        match Self::get_paths(control_flow_graph) {
            Some(paths) => {
                println!("=== CONTROL FLOW GRAPH PATHS ===");
                println!("Total paths found: {}", paths.len());
                println!(); // Added for spacing

                // Used `enumerate` for cleaner indexing.
                for (i, path) in paths.iter().enumerate() {
                    // Consider using a more detailed path printing format if needed for debugging.
                    println!("Path {}: {:?}", i, path);
                }
                println!(); // Added for spacing
            }
            None => {
                println!("No paths found in the control flow graph.");
            }
        }

        // Print the control flow graph nodes
        println!("=== CONTROL FLOW GRAPH NODES ===");
        // Iterate directly over the graph's items for clarity.
        for (node_id, node) in control_flow_graph.get_graph().iter() {
            println!("Node ID: {}, Node: {:#?}", node_id, node);
        }
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

        while let Some((current_node_id, mut path_so_far)) = stack.pop() { // path_so_far is now mutable
            // Get the current node from the graph.
            if let Some(node) = graph.get(&current_node_id) {
                match node {
                    Node::Cond { expr: _, stmts: _, succ } => { // stmts and expr are not used in this traversal logic
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
