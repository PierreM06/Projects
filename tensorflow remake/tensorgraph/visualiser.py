from pyvis.network import Network
from .tensor import Tensor
from typing import Any
import uuid
import json

def graph(tensor):
    graph = {
        "visualizations": {
            "computational_graph": tensor_to_graph(tensor)
        }
    }

    with open("graph.json", "w") as file:
        json.dump(graph, file, indent=2)

def tensor_to_graph(tensor: Tensor):
    visited_nodes = set()
    visited_edges = set()
    nodes, edges = _nodes_edges(tensor, visited_nodes, visited_edges)

    graph_dict = {
        "direction": "forwards",
        "nodes": nodes,
        "edges": edges
    }
    
    return graph_dict

def _nodes_edges(tensor: Tensor, visited_nodes: set[int], visited_edges: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = []
    edges = []
    node_id = id(tensor)

    if node_id in visited_nodes:
        return nodes, edges
    visited_nodes.add(node_id)

    node = {
        "id": node_id,
        "label": tensor._name or tensor._op or f"Tensor_{node_id}",
        "type": "operation",
        "right_panel": {
            "op_type": tensor._op,
            "name": tensor._name,
            "data": tensor.tolist(),
            "shape": list(tensor.data.shape),
            "dtype": str(tensor.data.dtype),
            "requires_grad": tensor.requires_grad,
            "gradient": tensor.grad.tolist() if tensor.grad is not None else None,
            # "extra": tensor._extra
        }
    }
    nodes.append(node)


    for parent in tensor._parents:
        parent_id = id(parent)
        edge_id = f"{node_id}_{parent_id}"
        parent_grad = tensor._parent_grads.get(parent_id, None)
        if edge_id not in visited_edges:
            edge = {
                "id": edge_id,
                "from": parent_id,
                "to": node_id,
                "label": "",
                "gradient": parent_grad.tolist() if parent_grad is not None else None,
                "right_panel": {
                    "from": parent_id,
                    "to": node_id,
                    "shape": list(parent_grad.shape) if parent_grad is not None else None,
                    "gradient": parent_grad.tolist() if parent_grad is not None else None,
                }
            }
            edges.append(edge)
            visited_edges.add(edge_id)

        sub_nodes, sub_edges = _nodes_edges(parent, visited_nodes, visited_edges)
        nodes.extend(sub_nodes)
        edges.extend(sub_edges)
    return nodes, edges


def get_tensor_label(tensor, fallback_id):
    return (
        tensor._name
        or tensor._op
        or fallback_id
    )

def get_tensor_tooltip(tensor):
    return (
        f"Name: {get_tensor_label(tensor, f"Tensor-{id(tensor) % 1000}")}\n"
        f"Data: {tensor.data}\n"
        f"Grad: {tensor.grad if tensor.grad is not None else 'None'}\n"
        f"Op: {tensor._op}\n"
        f"Shape: {tuple(tensor.data.shape)}\n"
        f"Grad shape: {tuple(tensor.grad.shape) if tensor.grad is not None else 'None'}\n"
        f"Extra: {tensor._extra}\n"
    )

def remove_pyvis_border(html: str) -> str:
    return html.replace("border: 1px solid lightgray;", "")

def visualize_tensor_graph(output_tensor, max_depth=10, output_file="pyvis_graph.html"):
    net = Network(height="100%", width="100%", directed=True)
    visited = set()

    def add_tensor_node(tensor, depth=0):
        if depth > max_depth or id(tensor) in visited:
            return
        visited.add(id(tensor))

        # First, ensure all parents are added to the graph
        for parent in tensor._parents:
            add_tensor_node(parent, depth + 1)

        # Then add the current tensor node
        label = get_tensor_label(tensor, f"Tensor-{id(tensor) % 1000}")
        tooltip = get_tensor_tooltip(tensor)

        net.add_node(id(tensor), label=label, title=tooltip, shape="box", color="#AED6F1")

        # Finally, add edges from the current tensor to its parents
        for parent in tensor._parents:
            edge_grad = tensor._parent_grads.get(id(parent), None)
            grad_str = str(edge_grad) if edge_grad is not None else "?"
            
            net.add_edge(
                id(parent),
                id(tensor),
                title=f"Gradient to parent:\n{grad_str}"
            )

    add_tensor_node(output_tensor)
    html = net.generate_html()
    html = remove_pyvis_border(html)
    with open("pyvis_graph.html", "w") as f:
        f.write(html)
    net.write_html(output_file)
