use std::collections::HashMap;

use crate::cfg::{ControlFlowGraph, Edge, Node, NodeId};

pub struct PathScraper;

impl PathScraper {
    /// Returns all paths in the control flow graph as a vector of vectors.
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

    /// Prints all paths in the control flow graph in a readable format.
    pub fn print_paths(control_flow_graph: &ControlFlowGraph) {
        match Self::get_paths(control_flow_graph) {
            Some(paths) => {
                println!("=== CONTROL FLOW GRAPH PATHS ===");
                println!("Total paths found: {}", paths.len());
                println!();

                for (i, path) in paths.iter().enumerate() {
                    println!("Path {}: {:?}", i, path);
                }
                println!();
            }
            None => {
                println!("No paths found in the control flow graph.");
            }
        }

        // Print the control flow graph nodes
        println!("=== CONTROL FLOW GRAPH NODES ===");
        for (node_id, node) in control_flow_graph.get_graph() {
            println!("Node ID: {}, Node: {:#?}", node_id, node);
        }
    }

    /// Depth-first collection of all root-to-leaf paths.
    fn traverse(graph: &HashMap<NodeId, Node>, entry: NodeId) -> Vec<Vec<(NodeId, Edge)>> {
        let mut all_paths: Vec<Vec<(NodeId, Edge)>> = Vec::new();
        let mut stack: Vec<(NodeId, Vec<(NodeId, Edge)>)> = Vec::new();

        stack.push((entry, Vec::new()));

        while let Some((current_node_id, path_so_far)) = stack.pop() {
            if let Some(node) = graph.get(&current_node_id) {
                match node {
                    Node::Cond { succ, .. } => {
                        let mut path = path_so_far.clone();
                        path.push((current_node_id, Edge::False));
                        stack.push((succ[1], path));

                        let mut path = path_so_far;
                        path.push((current_node_id, Edge::True));
                        stack.push((succ[0], path));
                    }
                    Node::Return { .. } | Node::Raise { .. } => {
                        let mut completed_path = path_so_far;
                        completed_path.push((current_node_id, Edge::Terminal));
                        all_paths.push(completed_path);
                    }
                }
            }
        }
        all_paths
    }
}
