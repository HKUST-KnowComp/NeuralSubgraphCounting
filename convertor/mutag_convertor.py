import igraph as ig
import os
import sys
from tqdm import tqdm

if __name__ == "__main__":
    assert len(sys.argv) == 2
    mutag_data_path = sys.argv[1]

    nodes = list()
    node2graph = list()
    with open(os.path.join(mutag_data_path, "MUTAG_graph_indicator.txt"), "r") as f:
        for n_id, g_id in enumerate(f):
            g_id = int(g_id)-1
            node2graph.append(g_id)
            if g_id == len(nodes):
                nodes.append(list())
            nodes[-1].append(n_id)

    nodelabels = [list() for _ in range(len(nodes))]
    with open(os.path.join(mutag_data_path, "MUTAG_node_labels.txt"), "r") as f:
        _nodelabels = list()
        for nl in f:
            nl = int(nl)
            _nodelabels.append(nl)
        n_idx = 0
        for g_idx in range(len(nodes)):
            for _ in range(len(nodes[g_idx])):
                nodelabels[g_idx].append(_nodelabels[n_idx])
                n_idx += 1
        del _nodelabels

    edges = [list() for _ in range(len(nodes))]
    with open(os.path.join(mutag_data_path, "MUTAG_A.txt"), "r") as f:
        for e in f:
            e = [int(v)-1 for v in e.split(",")]
            g_id = node2graph[e[0]]
            edges[g_id].append((e[0]-nodes[g_id][0], e[1]-nodes[g_id][0]))

    edgelabels = [list() for _ in range(len(nodes))]
    with open(os.path.join(mutag_data_path, "MUTAG_edge_labels.txt"), "r") as f:
        _edgelabels = list()
        for el in f:
            el = int(el)
            _edgelabels.append(el)
        e_idx = 0
        for g_idx in range(len(edges)):
            for _ in range(len(edges[g_idx])):
                edgelabels[g_idx].append(_edgelabels[e_idx])
                e_idx += 1
        del _edgelabels

    os.makedirs(os.path.join(mutag_data_path, "raw"), exist_ok=True)
    for g_id in tqdm(range(len(nodes))):
        graph = ig.Graph(directed=True)
        vcount = len(nodes[g_id])
        vlabels = nodelabels[g_id]
        elabels = edgelabels[g_id]
        
        graph.add_vertices(vcount)
        graph.add_edges(edges[g_id])
        graph.vs["label"] = vlabels
        graph.es["label"] = elabels
        graph.es["key"] = [0] * len(edges[g_id])
    
        graph_id = "G_N%d_E%d_NL%d_EL%d_%d" % (
            vcount, len(edges[g_id]), max(vlabels)+1, max(elabels)+1, g_id)
        filename = os.path.join(mutag_data_path, "raw", graph_id)
        # nx.nx_pydot.write_dot(pattern, filename + ".dot")
        graph.write(filename + ".gml")
