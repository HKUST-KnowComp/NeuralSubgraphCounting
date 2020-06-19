import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import re
import os
import json
from torch.optim.lr_scheduler import LambdaLR
from collections import OrderedDict
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

##########################################################
################## Evaluation Functions ##################
##########################################################
def compute_mae(predict, count):
    error = np.absolute(predict-count)
    return error.mean()

def compute_rmse(predict, count):
    error = np.power(predict-count, 2)
    return np.power(error.mean(), 0.5)

def compute_p_r_f1(predict, count):
    p, r, f1, _ = precision_recall_fscore_support(predict, count, average="binary")
    return p, r, f1

def compute_tp(predict, count):
    true_count = count == 1
    true_pred = predict == 1
    true_pred_count = true_count * true_pred
    return np.count_nonzero(true_pred_count) / np.count_nonzero(true_count)

##########################################################
#################### Parsing Functions ###################
##########################################################
def parse_pattern_info(x):
    p = re.findall(r"N(\d+)_E(\d+)_NL(\d+)_EL(\d+)", x)[0]
    return {"V": int(p[0]), "E": int(p[1]), "VL": int(p[2]), "EL": int(p[3])}

def parse_graph_info(x):
    g = re.findall(r"N(\d+)_E(\d+)_NL(\d+)_EL(\d+)_A([\d\.]+)", x)[0]
    return {"V": int(g[0]), "E": int(g[1]), "VL": int(g[2]), "EL": int(g[3]), "alpha": float(g[4])}

##########################################################
######### Representation and Encoding Functions ##########
##########################################################
def get_enc_len(x, base=10):
    # return math.floor(math.log(x, base)+1.0)
    l = 0
    while x:
        l += 1
        x = x // base
    return l

def int2onehot(x, len_x, base=10):
    if isinstance(x, (int, list)):
        x = np.array(x)
    x_shape = x.shape
    x = x.reshape(-1)
    one_hot = np.zeros((len_x*base, x.shape[0]), dtype=np.float32)
    x =  x % (base**len_x)
    idx = one_hot.shape[0] - base
    while np.any(x):
        x, y = x//base, x%base
        cond = y.reshape(1, -1) == np.arange(0, base, dtype=y.dtype).reshape(base, 1)
        one_hot[idx:idx+base] = np.where(cond, 1.0, 0.0)
        idx -= base
    while idx >= 0:
        one_hot[idx] = 1.0
        idx -= base
    one_hot = one_hot.transpose(1, 0).reshape(*x_shape, len_x*base)
    return one_hot

##########################################################
################ Deep Learning Functions #################
##########################################################
def segment_data(data, max_len):
    bsz = data.size(0)
    pad_len = max_len - data.size(1) % max_len
    if pad_len != max_len:
        pad_size = list(data.size())
        pad_size[1] = pad_len
        zero_pad = torch.zeros(pad_size, device=data.device, dtype=data.dtype, requires_grad=False)
        data = torch.cat([data, zero_pad], dim=1)
    return torch.split(data, max_len, dim=1)

def segment_length(data_len, max_len):
    bsz = data_len.size(0)
    list_len = math.ceil(data_len.max().float()/max_len)
    segment_lens = torch.arange(0, max_len*list_len, max_len, dtype=data_len.dtype, device=data_len.device, requires_grad=False).view(1, list_len)
    diff = data_len.view(-1, 1) - segment_lens
    fill_max = diff > max_len
    fill_zero = diff < 0
    segment_lens = diff.masked_fill(fill_max, max_len)
    segment_lens.masked_fill_(fill_zero, 0)
    return torch.split(segment_lens.view(bsz, -1), 1, dim=1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):
    bsz = graph_sizes.size(0)
    dim, dtype, device = batched_graph_feats.size(-1), batched_graph_feats.dtype, batched_graph_feats.device

    min_size, max_size = graph_sizes.min(), graph_sizes.max()
    mask = torch.ones((bsz, max_size), dtype=torch.uint8, device=device, requires_grad=False)

    if min_size == max_size:
        return batched_graph_feats.view(bsz, max_size, -1), mask
    else:
        graph_sizes_list = graph_sizes.view(-1).tolist()
        unbatched_graph_feats = list(torch.split(batched_graph_feats, graph_sizes_list, dim=0))
        for i, l in enumerate(graph_sizes_list):
            if l == max_size:
                continue
            elif l > max_size:
                unbatched_graph_feats[i] = unbatched_graph_feats[i][:max_size]
            else:
                mask[i, l:].fill_(0)
                zeros = torch.zeros((max_size-l, dim), dtype=dtype, device=device, requires_grad=False)
                unbatched_graph_feats[i] = torch.cat([unbatched_graph_feats[i], zeros], dim=0)
        return torch.stack(unbatched_graph_feats, dim=0), mask

def gather_indices_by_lens(lens):
    result = list()
    i, j = 0, 1
    max_j = len(lens)
    indices = np.arange(0, max_j)
    while j < max_j:
        if lens[i] != lens[j]:
            result.append(indices[i:j])
            i = j
        j += 1
    if i != j:
        result.append(indices[i:j])
    return result

def batch_convert_array_to_array(batch_array, max_seq_len=-1):
    batch_lens = [v.shape[0] for v in batch_array]
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)
    result = np.zeros([len(batch_array), max_seq_len] + list(batch_array[0].shape)[1:], dtype=batch_array[0].dtype)
    for i, t in enumerate(batch_array):
        len_t = batch_lens[i]
        if len_t < max_seq_len:
            result[i, :len_t] = t
        elif len_t == max_seq_len:
            result[i] = t
        else:
            result[i] = t[:max_seq_len]
    return result

def batch_convert_tensor_to_tensor(batch_tensor, max_seq_len=-1):
    batch_lens = [v.shape[0] for v in batch_tensor]
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)
    result = torch.zeros([len(batch_tensor), max_seq_len] + list(batch_tensor[0].size())[1:], dtype=batch_tensor[0].dtype, requires_grad=False)
    for i, t in enumerate(batch_tensor):
        len_t = batch_lens[i]
        if len_t < max_seq_len:
            result[i, :len_t].data.copy_(t)
        elif len_t == max_seq_len:
            result[i].data.copy_(t)
        else:
            result[i].data.copy_(t[:max_seq_len])
    return result

def batch_convert_len_to_mask(batch_lens, max_seq_len=-1):
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)
    mask = torch.ones((len(batch_lens), max_seq_len), dtype=torch.uint8, device=batch_lens[0].device, requires_grad=False)
    for i, l in enumerate(batch_lens):
        mask[i, l:].fill_(0)
    return mask

def convert_dgl_graph_to_edgeseq(graph, x_emb, x_len, e_emb):
    uid, vid, eid = graph.all_edges(form="all", order="srcdst")
    e = e_emb[eid]
    if x_emb is not None:
        u, v = x_emb[uid], x_emb[vid]
        e = torch.cat([u, v, e], dim=1)
    e_len = torch.tensor(graph.batch_num_edges, dtype=x_len.dtype, device=x_len.device).view(x_len.size())
    return e, e_len

def mask_seq_by_len(x, len_x):
    x_size = list(x.size())
    if x_size[1] == len_x.max():
        mask = batch_convert_len_to_mask(len_x)
        mask_size = x_size[0:2] + [1]*(len(x_size)-2)
        x = x * mask.view(*mask_size)
    return x

def extend_dimensions(old_layer, new_input_dim=-1, new_output_dim=-1, upper=False):
    if isinstance(old_layer, nn.Linear):
        old_output_dim, old_input_dim = old_layer.weight.size()
        if new_input_dim == -1:
            new_input_dim = old_input_dim
        if new_output_dim == -1:
            new_output_dim = old_output_dim
        assert new_input_dim >= old_input_dim and new_output_dim >= old_output_dim

        if new_input_dim != old_input_dim or new_output_dim != old_output_dim:
            use_bias = old_layer.bias is not None
            new_layer = nn.Linear(new_input_dim, new_output_dim, bias=use_bias)
            with torch.no_grad():
                nn.init.zeros_(new_layer.weight)
                if upper:
                    new_layer.weight[:old_output_dim, :old_input_dim].data.copy_(old_layer.weight)
                else:
                    new_layer.weight[-old_output_dim:, -old_input_dim:].data.copy_(old_layer.weight)
                if use_bias:
                    nn.init.zeros_(new_layer.bias)
                    if upper:
                        new_layer.bias[:old_output_dim].data.copy_(old_layer.bias)
                    else:
                        new_layer.bias[-old_output_dim:].data.copy_(old_layer.bias)
        else:
            new_layer = old_layer
    elif isinstance(old_layer, nn.LayerNorm):
        old_input_dim = old_layer.normalized_shape
        if len(old_input_dim) != 1:
            raise NotImplementedError
        old_input_dim = old_input_dim[0]
        assert new_input_dim >= old_input_dim
        if new_input_dim != old_input_dim and old_layer.elementwise_affine:
            new_layer = nn.LayerNorm(new_input_dim, elementwise_affine=True)
            with torch.no_grad():
                nn.init.ones_(new_layer.weight)
                nn.init.zeros_(new_layer.bias)
                if upper:
                    new_layer.weight[:old_input_dim].data.copy_(old_layer.weight)
                    new_layer.bias[:old_input_dim].data.copy_(old_layer.bias)
                else:
                    new_layer.weight[-old_input_dim:].data.copy_(old_layer.weight)
                    new_layer.bias[-old_input_dim:].data.copy_(old_layer.bias)
        else:
            new_layer = old_layer
    elif isinstance(old_layer, nn.LSTM):
        old_input_dim, old_output_dim = old_layer.input_size, old_layer.hidden_size
        if new_input_dim == -1:
            new_input_dim = old_input_dim
        if new_output_dim == -1:
            new_output_dim = old_output_dim
        assert new_input_dim >= old_input_dim and new_output_dim >= old_output_dim

        if new_input_dim != old_input_dim or new_output_dim != old_output_dim:
            new_layer = nn.LSTM(new_input_dim, new_output_dim,
                num_layers=old_layer.num_layers, bidirectional=old_layer.bidirectional,
                batch_first=old_layer.batch_first, bias=old_layer.bias)
            for layer_weights in new_layer._all_weights:
                for w in layer_weights:
                    with torch.no_grad():
                        if "weight" in w:
                            new_weight = getattr(new_layer, w)
                            old_weight = getattr(old_layer, w)
                            nn.init.zeros_(new_weight)
                            if upper:
                                new_weight[:old_weight.shape[0], :old_weight.shape[1]].data.copy_(old_weight)
                            else:
                                new_weight[-old_weight.shape[0]:, -old_weight.shape[1]:].data.copy_(old_weight)
                        if "bias" in w:
                            new_bias = getattr(new_layer, w)
                            old_bias = getattr(old_layer, w)
                            if new_bias is not None:
                                nn.init.zeros_(new_bias)
                                if upper:
                                    new_bias[:old_bias.shape[0]].data.copy_(old_bias)
                                else:
                                    new_bias[-old_bias.shape[0]:].data.copy_(old_bias)
    return new_layer


_act_map = {"none": lambda x: x,
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "softmax": nn.Softmax(dim=-1),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(1/5.5),
            "prelu": nn.PReLU(),
            "gelu": nn.GELU()}

def map_activation_str_to_layer(act_str):
    try:
        return _act_map[act_str]
    except:
        raise NotImplementedError("Error: %s activation fuction is not supported now." % (act_str))

def anneal_fn(fn, t, T, lambda0=0.0, lambda1=1.0):
    if not fn or fn == "none":
        return lambda1
    elif fn == "logistic":
        K = 8 / T
        return float(lambda0 + (lambda1-lambda0)/(1+np.exp(-K*(t-T/2))))
    elif fn == "linear":
        return float(lambda0 + (lambda1-lambda0) * t/T)
    elif fn == "cosine":
        return float(lambda0 + (lambda1-lambda0) * (1 - math.cos(math.pi * t/T))/2)
    elif fn.startswith("cyclical"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R*T, lambda0, lambda1)
        else:
            return anneal_fn(fn.split("_", 1)[1], t-R*T, R*T, lambda1, lambda0)
    elif fn.startswith("anneal"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R*T, lambda0, lambda1)
        else:
            return lambda1
    else:
        raise NotImplementedError

def get_constant_schedule(optimizer, last_epoch=-1):
    """ Create a schedule with a constant learning rate.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, min_percent=0.0):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(min_percent, float(num_training_steps - current_step) / float(max(1.0, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1, min_percent=0.0):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_percent, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=1.0, last_epoch=-1, min_percent=0.0):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function with several hard restarts, after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return min_percent
        return max(min_percent, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

##############################################
############ OS Function Parts ###############
##############################################
def _get_subdirs(dirpath, leaf_only=True):
    subdirs = list()
    is_leaf = True
    for filename in os.listdir(dirpath):
        if os.path.isdir(os.path.join(dirpath, filename)):
            is_leaf = False
            subdirs.extend(_get_subdirs(os.path.join(dirpath, filename), leaf_only=leaf_only))
    if not leaf_only or is_leaf:
        subdirs.append(dirpath)
    return subdirs

def _read_graphs_from_dir(dirpath):
    import igraph as ig
    graphs = dict()
    for filename in os.listdir(dirpath):
        if not os.path.isdir(os.path.join(dirpath, filename)):
            names = os.path.splitext(os.path.basename(filename))
            if names[1] != ".gml":
                continue
            try:
                graph = ig.read(os.path.join(dirpath, filename))
                graph.vs["label"] = [int(x) for x in graph.vs["label"]]
                graph.es["label"] = [int(x) for x in graph.es["label"]]
                graph.es["key"] = [int(x) for x in graph.es["key"]]
                graphs[names[0]] = graph
            except BaseException as e:
                print(e)
                break
    return graphs

def read_graphs_from_dir(dirpath, num_workers=4):
    graphs = dict()
    subdirs = _get_subdirs(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for subdir in subdirs:
            results.append((subdir, pool.apply_async(_read_graphs_from_dir, args=(subdir, ))))
        pool.close()
        
        for subdir, x in tqdm(results):
            x = x.get()
            graphs[os.path.basename(subdir)] = x
    return graphs

def read_patterns_from_dir(dirpath, num_workers=4):
    patterns = dict()
    subdirs = _get_subdirs(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for subdir in subdirs:
            results.append((subdir, pool.apply_async(_read_graphs_from_dir, args=(subdir, ))))
        pool.close()
        
        for subdir, x in tqdm(results):
            x = x.get()
            patterns.update(x)
    return patterns

def _read_metadata_from_dir(dirpath):
    meta = dict()
    for filename in os.listdir(dirpath):
        if not os.path.isdir(os.path.join(dirpath, filename)):
            names = os.path.splitext(os.path.basename(filename))
            if names[1] != ".meta":
                continue
            try:
                with open(os.path.join(dirpath, filename), "r") as f:
                    meta[names[0]] = json.load(f)
            except BaseException as e:
                print(e)
    return meta

def read_metadata_from_dir(dirpath, num_workers=4):
    meta = dict()
    subdirs = _get_subdirs(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for subdir in subdirs:
            results.append((subdir, pool.apply_async(_read_metadata_from_dir, args=(subdir, ))))
        pool.close()
        
        for subdir, x in tqdm(results):
            x = x.get()
            meta[os.path.basename(subdir)] = x
    return meta

def load_data(graph_dir, pattern_dir, metadata_dir, num_workers=4):
    patterns = read_patterns_from_dir(pattern_dir, num_workers=num_workers)
    graphs = read_graphs_from_dir(graph_dir, num_workers=num_workers)
    meta = read_metadata_from_dir(metadata_dir, num_workers=num_workers)

    train_data, dev_data, test_data = list(), list(), list()
    for p, pattern in patterns.items():
        if p in graphs:
            for g, graph in graphs[p].items():
                x = dict()
                x["id"] = ("%s-%s" % (p, g))
                x["pattern"] = pattern
                x["graph"] = graph
                x["subisomorphisms"] = meta[p][g]["subisomorphisms"]
                x["counts"] = meta[p][g]["counts"]

                g_idx = int(g.rsplit("_", 1)[-1])
                if g_idx % 10 == 0:
                    dev_data.append(x)
                elif g_idx % 10 == 1:
                    test_data.append(x)
                else:
                    train_data.append(x)
        elif len(graphs) == 1 and "raw" in graphs.keys():
            for g, graph in graphs["raw"].items():
                x = dict()
                x["id"] = ("%s-%s" % (p, g))
                x["pattern"] = pattern
                x["graph"] = graph
                x["subisomorphisms"] = meta[p][g]["subisomorphisms"]
                x["counts"] = meta[p][g]["counts"]

                g_idx = int(g.rsplit("_", 1)[-1])
                if g_idx % 3 == 0:
                    dev_data.append(x)
                elif g_idx % 3 == 1:
                    test_data.append(x)
                else:
                    train_data.append(x)
    return OrderedDict({"train": train_data, "dev": dev_data, "test": test_data})

def get_best_epochs(log_file):
    regex = re.compile(r"data_type:\s+(\w+)\s+best\s+([\s\w\-]+).*?\(epoch:\s+(\d+)\)")
    best_epochs = dict()
    # get the best epoch
    try:
        lines = subprocess.check_output(["tail", log_file, "-n3"]).decode("utf-8").split("\n")[0:-1]
        print(lines)
    except:
        with open(log_file, "r") as f:
            lines = f.readlines()
    
    for line in lines[-3:]:
        matched_results = regex.findall(line)
        for matched_result in matched_results:
            if "loss" in matched_result[1]:
                best_epochs[matched_result[0]] = int(matched_result[2])
    if len(best_epochs) != 3:
        for line in lines:
            matched_results = regex.findall(line)
            for matched_result in matched_results:
                if "loss" in matched_result[1]:
                    best_epochs[matched_result[0]] = int(matched_result[2])
    return best_epochs
