# import networkx as nx
import igraph as ig
import argparse
import numpy as np
import os
from utils import generate_labels, generate_tree, get_direction, str2bool
from collections import Counter, defaultdict
from time import time

def generate_patterns(number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels, number_of_patterns):
    patterns = []
    
    for p in range(number_of_patterns):
        start = time()
        
        pattern = ig.Graph(directed=True)
        
        # vertex labels
        vertex_labels = generate_labels(number_of_vertices, number_of_vertex_labels)
        # edge labels
        edge_labels = generate_labels(number_of_edges, number_of_edge_labels)

        # first, generate a tree
        pattern = generate_tree(number_of_vertices, directed=True)
        edge_label_mapping = defaultdict(set)
        for e, edge in enumerate(pattern.es):
            edge_label_mapping[edge.tuple].add(edge_labels[e])
        edge_keys = [0] * (number_of_vertices-1)

        # second, random add edges 
        ecount = pattern.ecount()
        new_edges = list()
        while ecount < number_of_edges:
            u = np.random.randint(0, number_of_vertices)
            v = np.random.randint(0, number_of_vertices)
            src_tgt = (u, v)
            edge_label = edge_labels[ecount]
            # # we do not generate edges between two same vertices with same labels
            if edge_label in edge_label_mapping[src_tgt]:
                continue
            new_edges.append(src_tgt)
            edge_keys.append(len(edge_label_mapping[src_tgt]))
            edge_label_mapping[src_tgt].add(edge_label)
            ecount += 1
        pattern.add_edges(new_edges)
        pattern.vs["label"] = vertex_labels
        pattern.es["label"] = edge_labels
        pattern.es["key"] = edge_keys

        patterns.append(pattern)
    return patterns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--number_of_vertices", type=int, default=3)
    parser.add_argument("--number_of_edges", type=int, default=3)
    parser.add_argument("--number_of_vertex_labels", type=int, default=2)
    parser.add_argument("--number_of_edge_labels", type=int, default=2)
    parser.add_argument("--number_of_patterns", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="patterns")
    parser.add_argument("--save_png", type=str2bool, default=False)
    args = parser.parse_args()

    np.random.seed(args.seed)

    patterns = generate_patterns(args.number_of_vertices, args.number_of_edges,
        args.number_of_vertex_labels, args.number_of_edge_labels,
        args.number_of_patterns)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for p, pattern in enumerate(patterns):
            pattern_id = "P_N%d_E%d_NL%d_EL%d_%d" % (
                args.number_of_vertices, args.number_of_edges, args.number_of_vertex_labels, args.number_of_edge_labels, p)
            filename = os.path.join(args.save_dir, pattern_id)
            # nx.nx_pydot.write_dot(pattern, filename + ".dot")
            pattern.write(filename + ".gml")
            if args.save_png:
                ig.plot(pattern, filename + ".png")

