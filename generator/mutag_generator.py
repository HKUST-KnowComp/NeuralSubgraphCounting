import igraph as ig
import numpy as np
import argparse
import os
import math
import json
from collections import Counter, defaultdict
from utils import generate_labels, generate_tree, get_direction, powerset, sample_element, str2bool, retrieve_multiple_edges
from pattern_checker import PatternChecker
from pattern_generator import generate_patterns
from graph_generator import GraphGenerator
from time import time

def generate_graphs(pattern, min_number_of_vertices, max_number_of_vertices, min_number_of_edges, max_number_of_edges, \
    min_number_of_vertex_labels, max_number_of_vertex_labels, min_number_of_edge_labels, max_number_of_edge_labels, \
    alpha, max_pattern_counts, max_subgraph, return_subisomorphisms, number_of_graphs):
    graph_generator = GraphGenerator(pattern)
    results = list()
    vl1, vl2 = math.log2(min_number_of_vertex_labels), math.log2(max_number_of_vertex_labels)
    el1, el2 = math.log2(min_number_of_edge_labels), math.log2(max_number_of_edge_labels)
    for g in range(number_of_graphs):
        number_of_vertices = np.random.randint(min_number_of_vertices, max_number_of_vertices+1)
        number_of_edges = np.random.randint(max(min_number_of_edges, number_of_vertices), max_number_of_edges+1)
        number_of_vertex_labels = math.floor(math.pow(np.random.rand()*(vl2-vl1)+vl1, 2))
        number_of_edge_labels = math.floor(math.pow(np.random.rand()*(el2-el1)+el1, 2))
        graph, metadata = graph_generator.generate(
                number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels,
                alpha, max_pattern_counts=max_pattern_counts, max_subgraph=max_subgraph,
                return_subisomorphisms=return_subisomorphisms)
        print("%d/%d" % (g+1, number_of_graphs), "number of subisomorphisms: %d" % (metadata["counts"]))
        results.append((graph, metadata))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_number_of_vertices", type=int, default=10)
    parser.add_argument("--max_number_of_vertices", type=int, default=28)
    parser.add_argument("--min_number_of_edges", type=int, default=20//2)
    parser.add_argument("--max_number_of_edges", type=int, default=66//2)
    parser.add_argument("--min_number_of_vertex_labels", type=int, default=3)
    parser.add_argument("--max_number_of_vertex_labels", type=int, default=7)
    parser.add_argument("--min_number_of_edge_labels", type=int, default=3)
    parser.add_argument("--max_number_of_edge_labels", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--max_pattern_counts", type=float, default=128)
    parser.add_argument("--return_subisomorphisms", type=str2bool, default=True)
    parser.add_argument("--max_subgraph", type=int, default=512)
    parser.add_argument("--number_of_graphs", type=int, default=56)
    parser.add_argument("--pattern_path", type=str,default=r"patterns/P_N3_E3_NL2_EL2_0.gml")
    parser.add_argument("--save_graph_dir", type=str, default="graphs")
    parser.add_argument("--save_metadata_dir", type=str, default="metadata")
    parser.add_argument("--save_png", type=str2bool, default=False)
    parser.add_argument("--show_img", type=str2bool, default=False)
    args = parser.parse_args()

    np.random.seed(args.seed)

    try:
        pattern = ig.read(args.pattern_path)
        pattern.vs["label"] = [int(x) for x in pattern.vs["label"]]
        pattern.es["label"] = [int(x) for x in pattern.es["label"]]
        pattern.es["key"] = [int(x) for x in pattern.es["key"]]
    except BaseException as e:
        print(e)
        pattern = ig.Graph(directed=True)
        pattern.vs["label"] = []
        pattern.es["label"] = []
        pattern.es["key"] = []

    results = generate_graphs(pattern,
        args.min_number_of_vertices, args.max_number_of_vertices,
        args.min_number_of_edges, args.max_number_of_edges,
        args.min_number_of_vertex_labels, args.max_number_of_vertex_labels,
        args.min_number_of_edge_labels, args.max_number_of_edge_labels,
        args.alpha, args.max_pattern_counts, args.max_subgraph,
        args.return_subisomorphisms, args.number_of_graphs)

    if args.save_graph_dir:
        os.makedirs(args.save_graph_dir, exist_ok=True)
        save_graph_dir = os.path.join(args.save_graph_dir, os.path.splitext(os.path.basename(args.pattern_path))[0])
        os.makedirs(save_graph_dir, exist_ok=True)
        if args.save_metadata_dir:
            os.makedirs(args.save_metadata_dir, exist_ok=True)
            save_metadata_dir = os.path.join(args.save_metadata_dir, os.path.splitext(os.path.basename(args.pattern_path))[0])
            os.makedirs(save_metadata_dir, exist_ok=True)
        for g, (graph, metadata) in enumerate(results):
            graph_id = "G_N%d_E%d_NL%d_EL%d_%d" % (
                graph.vcount(), graph.ecount(), max(graph.vs["label"])+1, max(graph.es["label"])+1, g)
            graph_filename = os.path.join(save_graph_dir, graph_id)
            graph.write(graph_filename + ".gml")
            if args.save_metadata_dir:
                metadata_filename = os.path.join(save_metadata_dir, graph_id)
                with open(metadata_filename + ".meta", "w") as f:
                    json.dump(metadata, f)
            if args.save_png:
                ig.plot(graph, graph_filename + ".png")
            if args.show_img:
                draw(graph, pattern, metadata["subisomorphisms"])
