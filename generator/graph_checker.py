import igraph as ig
import numpy as np
import argparse
import os
import math
import json
import shutil
from collections import Counter, defaultdict
from utils import generate_labels, generate_tree, get_direction, powerset, sample_element, str2bool, retrieve_multiple_edges
from pattern_checker import PatternChecker
from time import time
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool

def get_subisomorphisms(pattern, graphs):
    results = dict()
    pattern_checker = PatternChecker()
    for gid, graph in graphs.items():
        subisomorphisms = pattern_checker.get_subisomorphisms(graph, pattern)
        results[gid] = subisomorphisms
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern_dir", type=str, default=0)
    parser.add_argument("--raw_graph_dir", type=str, default=3)
    parser.add_argument("--save_graph_dir", type=str, default="graphs")
    parser.add_argument("--save_metadata_dir", type=str, default="metadata")
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    os.makedirs(os.path.join(args.save_graph_dir), exist_ok=True)
    os.makedirs(os.path.join(args.save_metadata_dir), exist_ok=True)
    with Pool(args.num_workers) as pool:
        pool_results = list()
        patterns = dict()
        graphs = dict()
        for pid in os.listdir(args.pattern_dir):
            if not pid.endswith(".gml"):
                continue
            pattern = ig.read(os.path.join(args.pattern_dir, pid))
            patterns[os.path.splitext(pid)[0]] = pattern
            # shutil.copytree(args.raw_graph_dir, os.path.join(args.save_graph_dir, os.path.splitext(pid)[0]))
            # os.system("ln -s %s %s" % (args.raw_graph_dir, os.path.join(args.save_graph_dir, os.path.splitext(pid)[0])))
        
        for gid in os.listdir(args.raw_graph_dir):
            if not gid.endswith(".gml"):
                continue
            graph = ig.read(os.path.join(args.raw_graph_dir, gid))
            graphs[os.path.splitext(gid)[0]] = graph

        for pid, pattern in patterns.items():
            pool_results.append((pid, pool.apply_async(get_subisomorphisms, args=(pattern, graphs))))
        for x in tqdm(pool_results):
            pid, x = x
            x = x.get()
            os.makedirs(os.path.join(args.save_metadata_dir, pid), exist_ok=True)
            
            mae = 0
            mse = 0
            for gid, subisomorphisms in x.items():
                mae += len(subisomorphisms)
                mse += len(subisomorphisms) * len(subisomorphisms)
                with open(os.path.join(args.save_metadata_dir, pid, gid+".meta"), "w") as f:
                    json.dump({"counts": len(subisomorphisms), "subisomorphisms": subisomorphisms}, f)
            print("pid", "mae", mae/len(x), "mse", mse/len(x))

