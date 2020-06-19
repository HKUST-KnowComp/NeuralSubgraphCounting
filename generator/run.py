import numpy as np
import argparse
import os
import igraph as ig
import json
from multiprocessing import Pool
from utils import generate_labels, get_direction
from pattern_checker import PatternChecker
from graph_generator import GraphGenerator
from pattern_generator import generate_patterns
from time import sleep
from tqdm import tqdm


DEBUG_CONFIG = {
    "max_subgraph": 512,

    "alphas": [0.5],

    "number_of_patterns": 1,
    "number_of_pattern_vertices": [3, 4],
    "number_of_pattern_edges": [2, 4],
    "number_of_pattern_vertex_labels": [2, 4],
    "number_of_pattern_edge_labels": [2, 4],

    "number_of_graphs": 10, # train:dev:test = 8:1:1
    "number_of_graph_vertices": [16, 64],
    "number_of_graph_edges": [16, 64, 256],
    "number_of_graph_vertex_labels": [4, 8],
    "number_of_graph_edge_labels": [4, 8],

    "max_ratio_of_edges_vertices": 4,
    "max_pattern_counts": 1024,

    "save_data_dir": r"../data/debug",
    "num_workers": 16
}

SMALL_CONFIG = {
    "max_subgraph": 512,

    "alphas": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],

    "number_of_patterns": 3,
    "number_of_pattern_vertices": [3, 4, 8],
    "number_of_pattern_edges": [2, 4, 8],
    "number_of_pattern_vertex_labels": [2, 4, 8],
    "number_of_pattern_edge_labels": [2, 4, 8],

    "number_of_graphs": 10, # train:dev:test = 8:1:1
    "number_of_graph_vertices": [8, 16, 32, 64],
    "number_of_graph_edges": [8, 16, 32, 64, 128, 256],
    "number_of_graph_vertex_labels": [4, 8, 16],
    "number_of_graph_edge_labels": [4, 8, 16],

    "max_ratio_of_edges_vertices": 4,
    "max_pattern_counts": 1024,

    "save_data_dir": r"/data/xliucr/SubIsoCnt/small",
    "num_workers": 16
}

LARGE_CONFIG = {
    "max_subgraph": 512,

    "alphas": [0.05, 0.1, 0.15],
    
    "number_of_patterns": 2,
    "number_of_pattern_vertices": [3, 4, 8, 16],
    "number_of_pattern_edges": [2, 4, 8, 16],
    "number_of_pattern_vertex_labels": [2, 4, 8, 16],
    "number_of_pattern_edge_labels": [2, 4, 8, 16],

    "number_of_graphs": 10, # train:dev:test = 8:1:1
    "number_of_graph_vertices": [64, 128, 256, 512],
    "number_of_graph_edges": [64, 128, 256, 512, 1024, 2048],
    "number_of_graph_vertex_labels": [16, 32, 64],
    "number_of_graph_edge_labels": [16, 32, 64],

    "max_ratio_of_edges_vertices": 4,
    "max_pattern_counts": 4096,

    "save_data_dir": r"/data/xliucr/SubIsoCnt/large",
    "num_workers": 16
}

CONFIG = DEBUG_CONFIG

def generate_graphs(graph_generator, number_of_graph_vertices, number_of_graph_edges, number_of_graph_vertex_labels, number_of_graph_edge_labels,
    alpha, max_pattern_counts, max_subgraph, number_of_graphs, save_graph_dir, save_metadata_dir):
    graphs_id = "G_N%d_E%d_NL%d_EL%d_A%.2f" % (
        number_of_graph_vertices, number_of_graph_edges, number_of_graph_vertex_labels, number_of_graph_edge_labels, alpha)
    # print(graphs_id)
    for g in range(number_of_graphs):
        graph, metadata = graph_generator.generate( 
            number_of_graph_vertices, number_of_graph_edges, number_of_graph_vertex_labels, number_of_graph_edge_labels,
            alpha, max_pattern_counts=max_pattern_counts, max_subgraph=max_subgraph, return_subisomorphisms=True)
        graph.write(os.path.join(save_graph_dir, graphs_id + "_%d.gml" % (g)))
        with open(os.path.join(save_metadata_dir, graphs_id + "_%d.meta" % (g)), "w") as f:
            json.dump(metadata, f)
    return graphs_id

if __name__ == "__main__":
    save_pattern_dir = os.path.join(CONFIG["save_data_dir"], "patterns")
    save_graph_dir = os.path.join(CONFIG["save_data_dir"], "graphs")
    save_metadata_dir = os.path.join(CONFIG["save_data_dir"], "metadata")
    os.makedirs(CONFIG["save_data_dir"], exist_ok=True)
    os.makedirs(save_pattern_dir, exist_ok=True)
    os.makedirs(save_graph_dir, exist_ok=True)
    os.makedirs(save_metadata_dir, exist_ok=True)

    np.random.seed(0)

    pattern_cnt = 0
    for number_of_pattern_vertices in CONFIG["number_of_pattern_vertices"]:
        for number_of_pattern_vertex_labels in CONFIG["number_of_pattern_vertex_labels"]:
            if number_of_pattern_vertex_labels > number_of_pattern_vertices:
                continue
            for number_of_pattern_edges in CONFIG["number_of_pattern_edges"]:
                if number_of_pattern_edges < number_of_pattern_vertices - 1: # not connected
                    continue
                if number_of_pattern_edges > CONFIG["max_ratio_of_edges_vertices"] * number_of_pattern_vertices: # too dense
                    continue
                for number_of_pattern_edge_labels in CONFIG["number_of_pattern_edge_labels"]:
                    if number_of_pattern_edge_labels > number_of_pattern_edges:
                        continue
                    patterns_id = "P_N%d_E%d_NL%d_EL%d" % (
                        number_of_pattern_vertices, number_of_pattern_edges, number_of_pattern_vertex_labels, number_of_pattern_edge_labels)
                    for p, pattern in enumerate(generate_patterns(
                        number_of_pattern_vertices, number_of_pattern_edges, number_of_pattern_vertex_labels, number_of_pattern_edge_labels,
                        CONFIG["number_of_patterns"])):
                        pattern.write(os.path.join(save_pattern_dir, patterns_id + "_%d.gml" % (p)))
                    pattern_cnt += CONFIG["number_of_patterns"]
                    print("patterns_id", patterns_id)
    print("%d patterns generation finished!" % (pattern_cnt))

    graph_cnt = 0
    pool = Pool(CONFIG["num_workers"])
    results = list()
    for number_of_pattern_vertices in CONFIG["number_of_pattern_vertices"]:
        for number_of_pattern_vertex_labels in CONFIG["number_of_pattern_vertex_labels"]:
            if number_of_pattern_vertex_labels > number_of_pattern_vertices:
                continue
            for number_of_pattern_edges in CONFIG["number_of_pattern_edges"]:
                if number_of_pattern_edges < number_of_pattern_vertices - 1: # not connected
                    continue
                if number_of_pattern_edges > CONFIG["max_ratio_of_edges_vertices"] * number_of_pattern_vertices: # too dense
                    continue
                for number_of_pattern_edge_labels in CONFIG["number_of_pattern_edge_labels"]:
                    if number_of_pattern_edge_labels > number_of_pattern_edges:
                        continue
                    patterns_id = "P_N%d_E%d_NL%d_EL%d" % (
                        number_of_pattern_vertices, number_of_pattern_edges, number_of_pattern_vertex_labels, number_of_pattern_edge_labels)
                    graph_generators = list()
                    for p in range(CONFIG["number_of_patterns"]):
                        pattern = ig.read(os.path.join(save_pattern_dir, patterns_id + "_%d.gml" % (p)))
                        pattern.vs["label"] = [int(x) for x in pattern.vs["label"]]
                        pattern.es["label"] = [int(x) for x in pattern.es["label"]]
                        pattern.es["key"] = [int(x) for x in pattern.es["key"]]
                        graph_generators.append(GraphGenerator(pattern))
                    for alpha in CONFIG["alphas"]:
                        for number_of_graph_vertices in CONFIG["number_of_graph_vertices"]:
                            if number_of_graph_vertices < number_of_pattern_vertices:
                                continue
                            for number_of_graph_vertex_labels in CONFIG["number_of_graph_vertex_labels"]:
                                if number_of_graph_vertex_labels > number_of_graph_vertices:
                                    continue
                                if number_of_graph_vertex_labels < number_of_pattern_vertex_labels:
                                    continue
                                for number_of_graph_edges in CONFIG["number_of_graph_edges"]:
                                    if number_of_graph_edges < number_of_graph_vertices - 1: # not connected
                                        continue
                                    if number_of_graph_edges > CONFIG["max_ratio_of_edges_vertices"] * number_of_graph_vertices: # too dense
                                        continue
                                    if number_of_graph_edges < number_of_pattern_edges:
                                        continue
                                    for number_of_graph_edge_labels in CONFIG["number_of_graph_edge_labels"]:
                                        if number_of_graph_edge_labels > number_of_graph_edges:
                                            continue
                                        if number_of_graph_edge_labels < number_of_pattern_edge_labels:
                                            continue
                                        for p, graph_generator in enumerate(graph_generators):
                                            save_graph_dir_p = os.path.join(save_graph_dir, patterns_id + "_%d" % (p))
                                            save_metadata_dir_p = os.path.join(save_metadata_dir, patterns_id + "_%d" % (p))
                                            if not os.path.isdir(save_graph_dir_p):
                                                os.mkdir(save_graph_dir_p)
                                            if not os.path.isdir(save_metadata_dir_p):
                                                os.mkdir(save_metadata_dir_p)
                                            results.append(
                                                pool.apply_async(generate_graphs, args=(
                                                    graph_generator, number_of_graph_vertices, number_of_graph_edges,
                                                    number_of_graph_vertex_labels, number_of_graph_edge_labels,
                                                    alpha, CONFIG["max_pattern_counts"], CONFIG["max_subgraph"],
                                                    CONFIG["number_of_graphs"], save_graph_dir_p, save_metadata_dir_p)))
                                            # generate_graphs(
                                            #         graph_generator, number_of_graph_vertices, number_of_graph_edges,
                                            #         number_of_graph_vertex_labels, number_of_graph_edge_labels,
                                            #         alpha, CONFIG["max_pattern_counts"], CONFIG["max_subgraph"],
                                            #         CONFIG["number_of_graphs"], save_graph_dir_p, save_metadata_dir_p)
                                            graph_cnt += CONFIG["number_of_graphs"]
    pool.close()
    # pool.join()
    for x in tqdm(results):
        x.get()
    print("%d graphs generation finished!" % (graph_cnt))
