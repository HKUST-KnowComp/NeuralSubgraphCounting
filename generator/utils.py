import numpy as np
import igraph as ig
import json
from itertools import chain, combinations

def generate_png(dot_filename, png_filename=None, prog="neato"):
    if png_filename is None:
        png_filename = dot_filename.replace(".dot", ".png")
    os.system("%s.exe -T png %s > %s" % (prog, dot_filename, png_filename))

def generate_labels(number_of_items, number_of_labels):
    labels = list(range(number_of_labels))
    if number_of_items < number_of_labels:
        np.random.shuffle(labels)
        labels = labels[:number_of_items]
    else:
        for i in range(number_of_labels, number_of_items):
            labels.append(np.random.randint(number_of_labels))
        np.random.shuffle(labels)
    return labels

def generate_tree(number_of_vertices, directed=True):
    # Alexey S. Rodionov and Hyunseung Choo, On Generating Random Network Structures: Trees, ICCS 2003, LNCS 2658, pp. 879-887, 2003.
    # [connected vertices] + [unconnected vertices]
    shuffle_vertices = list(range(number_of_vertices))
    np.random.shuffle(shuffle_vertices)
    # randomly choose one vertex from the connected vertex set
    # randomly choose one vertex from the unconnected vertex set
    # connect them by one edge
    # add the latter vertex in the connected vertex set
    edges = list()
    for v in range(1, number_of_vertices):
        u = shuffle_vertices[np.random.randint(0, v)]
        v = shuffle_vertices[v]
        if get_direction():
            src_tgt = (u, v)
        else:
            src_tgt = (v, u)
        edges.append(src_tgt)
    tree = ig.Graph(directed=directed)
    tree.add_vertices(number_of_vertices)
    tree.add_edges(edges)
    return tree

def get_direction():
    return np.random.randint(0, 2)

def retrieve_multiple_edges(graph, source=-1, target=-1):
    if source != -1:
        e = graph.incident(source, mode=ig.OUT)
        if target != -1:
            e = set(e).intersection(graph.incident(target, mode=ig.IN))
        return ig.EdgeSeq(graph, e)     
    else:
        if target != -1:
            e = graph.incident(target, mode=ig.IN)
        else:
            e = list()
        return ig.EdgeSeq(graph, e)

def str2bool(x):
    x = x.lower()
    return x == "true" or x == "yes" or x == "t"

def sample_element(s):
    index = np.random.randint(0, len(s))
    return s[index]

def powerset(iterable, min_size=0, max_size=-1):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = sorted(iterable)
    if max_size == -1:
        max_size = len(s)
    return chain.from_iterable(combinations(s, r) for r in range(min_size, max_size+1))