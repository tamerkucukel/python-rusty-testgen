use std::collections::HashMap;

use crate::cfg::{ControlFlowGraph, Edge, Node, NodeId};

pub struct Explorer;

impl Explorer {
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

    /// Depth-first collection of all root-to-leaf paths.
    fn traverse(graph: &HashMap<NodeId, Node>, entry: NodeId) -> Vec<Vec<(NodeId, Edge)>> {
        let mut paths = Vec::new();

        let mut stack: Vec<(NodeId, Vec<(NodeId, Edge)>)> =
            vec![(entry, vec![(entry, Edge::True)])];

        while let Some((node_id, path_so_far)) = stack.pop() {
            match graph.get(&node_id) {
                Some(Node::Cond { succ, .. }) => {
                    // ---- FALSE branch first (so TRUE is popped/visited first) ----
                    let mut path_false = path_so_far.clone();
                    path_false.push((succ[1], Edge::False));
                    stack.push((succ[1], path_false));

                    // ---- TRUE branch ----
                    let mut path_true = path_so_far;
                    path_true.push((succ[0], Edge::True));
                    stack.push((succ[0], path_true));
                }

                // Any terminal node ends a path.
                Some(Node::Return { .. }) | Some(Node::Raise { .. }) => {
                    paths.push(path_so_far);
                }

                _ => {} // unreachable / malformed graph entry
            }
        }

        paths
    }
}
