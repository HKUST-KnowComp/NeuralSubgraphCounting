import torch
import os
import numpy as np
import dgl
import logging
import datetime
import math
import sys
import gc
import json
import time
import torch.nn.functional as F
import warnings
from functools import partial
from collections import OrderedDict
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except BaseException as e:
    from tensorboardX import SummaryWriter
from dataset import Sampler, EdgeSeqDataset, GraphAdjDataset
from utils import anneal_fn, get_enc_len, load_data, get_linear_schedule_with_warmup
from cnn import CNN
from rnn import RNN
from txl import TXL
from rgcn import RGCN
from rgin import RGIN
from train import evaluate, train

warnings.filterwarnings("ignore")
INF = float("inf")

train_config = {
    "max_npv": 4, # max_number_pattern_vertices: 8, 16, 32
    "max_npe": 3, # max_number_pattern_edges: 8, 16, 32
    "max_npvl": 2, # max_number_pattern_vertex_labels: 8, 16, 32
    "max_npel": 2, # max_number_pattern_edge_labels: 8, 16, 32

    "max_ngv": 28, # max_number_graph_vertices: 64, 512,4096
    "max_nge": 66, # max_number_graph_edges: 256, 2048, 16384
    "max_ngvl": 7, # max_number_graph_vertex_labels: 16, 64, 256
    "max_ngel": 4, # max_number_graph_edge_labels: 16, 64, 256
    
    "base": 2,

    "gpu_id": -1,
    "num_workers": 12,
    
    "epochs": 100,
    "batch_size": 64,
    "update_every": 1, # actual batch_sizer = batch_size * update_every
    "print_every": 100,
    "init_emb": "Equivariant", # None, Orthogonal, Normal, Equivariant
    "share_emb": True, # sharing embedding requires the same vector length
    "share_arch": True, # sharing architectures
    "dropout": 0.2,
    "dropatt": 0.2,
    
    "reg_loss": "MSE", # MAE, MSEl
    "bp_loss": "MSE", # MAE, MSE
    "bp_loss_slp": "anneal_cosine$1.0$0.01",    # 0, 0.01, logistic$1.0$0.01, linear$1.0$0.01, cosine$1.0$0.01, 
                                                # cyclical_logistic$1.0$0.01, cyclical_linear$1.0$0.01, cyclical_cosine$1.0$0.01
                                                # anneal_logistic$1.0$0.01, anneal_linear$1.0$0.01, anneal_cosine$1.0$0.01
    "lr": 0.001,
    "weight_decay": 0.00001,
    "max_grad_norm": 8,

    "model" : "CNN", # CNN, RNN, TXL, RGCN, RGIN, RSIN
    
    "emb_dim": 128,
    "activation_function": "leaky_relu", # sigmoid, softmax, tanh, relu, leaky_relu, prelu, gelu

    "filter_net": "MaxGatedFilterNet", # None, MaxGatedFilterNet
    "predict_net": "SumPredictNet", # MeanPredictNet, SumPredictNet, MaxPredictNet,
                                    # MeanAttnPredictNet, SumAttnPredictNet, MaxAttnPredictNet,
                                    # MeanMemAttnPredictNet, SumMemAttnPredictNet, MaxMemAttnPredictNet,
                                    # DIAMNet
    "predict_net_add_enc": True,
    "predict_net_add_degree": True,
    "predict_net_hidden_dim": 128,
    "predict_net_num_heads": 4,
    "predict_net_mem_len": 4,
    "predict_net_mem_init": "mean", # mean, sum, max, attn, circular_mean, circular_sum, circular_max, circular_attn, lstm
    "predict_net_recurrent_steps": 3,
    
    "cnn_hidden_dim": 128,
    "cnn_conv_channels": (128, 128, 128),
    "cnn_conv_kernel_sizes": (2, 3, 4),
    "cnn_conv_strides": (1, 1, 1),
    "cnn_conv_paddings": (0, 1, 1),
    "cnn_pool_kernel_sizes": (2, 3, 4),
    "cnn_pool_strides": (1, 1, 1),
    "cnn_pool_paddings": (1, 1, 2),
    
    "rnn_type": "LSTM", # GRU, LSTM
    "rnn_bidirectional": False,
    "rnn_graph_num_layers": 3,
    "rnn_pattern_num_layers": 3,
    "rnn_hidden_dim": 128,

    "txl_graph_num_layers": 3,
    "txl_pattern_num_layers": 3,
    "txl_d_model": 128,
    "txl_d_inner": 128,
    "txl_n_head": 4,
    "txl_d_head": 4,
    "txl_pre_lnorm": True,
    "txl_tgt_len": 64,
    "txl_ext_len": 0, # useless in current settings
    "txl_mem_len": 64,
    "txl_clamp_len": -1, # max positional embedding index
    "txl_attn_type": 0, # 0 for Dai et al, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.
    "txl_same_len": False,

    "rgcn_num_bases": 8,
    "rgcn_regularizer": "bdd", # basis, bdd
    "rgcn_graph_num_layers": 3,
    "rgcn_pattern_num_layers": 3,
    "rgcn_hidden_dim": 128,
    "rgcn_ignore_norm": False, # ignorm=True -> RGCN-SUM

    "rgin_num_bases": 8,
    "rgin_regularizer": "bdd", # basis, bdd
    "rgin_graph_num_layers": 3,
    "rgin_pattern_num_layers": 3,
    "rgin_hidden_dim": 128,
    
    "train_ratio": 1.0,
    "pattern_dir": "../data/MUTAG/patterns",
    "graph_dir": "../data/MUTAG/raw",
    "metadata_dir": "../data/MUTAG/metadata",
    "save_data_dir": "../data/MUTAG/",
    "save_model_dir": "../dumps/MUTAG",
}


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    for i in range(1, len(sys.argv), 2):
        arg = sys.argv[i]
        value = sys.argv[i+1]
        
        if arg.startswith("--"):
            arg = arg[2:]
        if arg not in train_config:
            print("Warning: %s is not surported now." % (arg))
            continue
        train_config[arg] = value
        try:
            value = eval(value)
            if isinstance(value, (int, float)):
                train_config[arg] = value
        except:
            pass

    ts = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_name = "%s_%s_%s" % (train_config["model"], train_config["predict_net"], ts)
    save_model_dir = train_config["save_model_dir"]
    os.makedirs(save_model_dir, exist_ok=True)
    
    # save config
    with open(os.path.join(save_model_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f)

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(os.path.join(save_model_dir, "train_log.txt"), 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    # set device
    device = torch.device("cuda:%d" % train_config["gpu_id"] if train_config["gpu_id"] != -1 else "cpu")
    if train_config["gpu_id"] != -1:
        torch.cuda.set_device(device)

    # reset the pattern parameters
    if train_config["share_emb"]:
        train_config["max_npv"], train_config["max_npvl"], train_config["max_npe"], train_config["max_npel"] = \
            train_config["max_ngv"], train_config["max_ngvl"], train_config["max_nge"], train_config["max_ngel"]

    # construct the model
    if train_config["model"] == "CNN":
        model = CNN(train_config)
    elif train_config["model"] == "RNN":
        model = RNN(train_config)
    elif train_config["model"] == "TXL":
        model = TXL(train_config)
    elif train_config["model"] == "RGCN":
        model = RGCN(train_config)
    elif train_config["model"] == "RGIN":
        model = RGIN(train_config)
    else:
        raise NotImplementedError("Currently, the %s model is not supported" % (train_config["model"]))

    model = model.to(device)
    logger.info(model)
    logger.info("num of parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # load data
    os.makedirs(train_config["save_data_dir"], exist_ok=True)
    data_loaders = OrderedDict({"train": None, "dev": None, "test": None})
    if all([os.path.exists(os.path.join(train_config["save_data_dir"],
        "%s_%s_dataset.pt" % (
            data_type, "dgl" if train_config["model"] in ["RGCN", "RGIN"] else "edgeseq"))) for data_type in data_loaders]):

        logger.info("loading data from pt...")
        for data_type in data_loaders:
            if train_config["model"] in ["RGCN", "RGIN"]:
                dataset = GraphAdjDataset(list())
                dataset.load(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                if data_type == "train":
                    np.random.shuffle(dataset.data)
                    dataset.data = dataset.data[:math.ceil(len(dataset.data)*train_config["train_ratio"])]
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=train_config["batch_size"], shuffle=data_type=="train", drop_last=False)
                data_loader = DataLoader(dataset,
                    batch_sampler=sampler,
                    collate_fn=GraphAdjDataset.batchify,
                    pin_memory=data_type=="train")
            else:
                dataset = EdgeSeqDataset(list())
                dataset.load(os.path.join(train_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type)))
                if data_type == "train":
                    np.random.shuffle(dataset.data)
                    dataset.data = dataset.data[:math.ceil(len(dataset.data)*train_config["train_ratio"])]
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=train_config["batch_size"], shuffle=data_type=="train", drop_last=False)
                data_loader = DataLoader(dataset,
                    batch_sampler=sampler,
                    collate_fn=EdgeSeqDataset.batchify,
                    pin_memory=data_type=="train")
            data_loaders[data_type] = data_loader
            logger.info("data (data_type: {:<5s}, len: {}) generated".format(data_type, len(dataset.data)))
            logger.info("data_loader (data_type: {:<5s}, len: {}, batch_size: {}) generated".format(data_type, len(data_loader), train_config["batch_size"]))
    else:
        data = load_data(train_config["graph_dir"], train_config["pattern_dir"], train_config["metadata_dir"], num_workers=train_config["num_workers"])
        logger.info("{}/{}/{} data loaded".format(len(data["train"]), len(data["dev"]), len(data["test"])))
        for data_type, x in data.items():
            if train_config["model"] in ["RGCN", "RGIN"]:
                if os.path.exists(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type))):
                    dataset = GraphAdjDataset(list())
                    dataset.load(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                else:
                    dataset = GraphAdjDataset(x)
                    dataset.save(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                if data_type == "train":
                    np.random.shuffle(dataset.data)
                    dataset.data = dataset.data[:math.ceil(len(dataset.data)*train_config["train_ratio"])]
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=train_config["batch_size"], shuffle=data_type=="train", drop_last=False)
                data_loader = DataLoader(dataset,
                    batch_sampler=sampler,
                    collate_fn=GraphAdjDataset.batchify,
                    pin_memory=data_type=="train")
            else:
                if os.path.exists(os.path.join(train_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type))):
                    dataset = EdgeSeqDataset(list())
                    dataset.load(os.path.join(train_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type)))
                else:
                    dataset = EdgeSeqDataset(x)
                    dataset.save(os.path.join(train_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type)))
                if data_type == "train":
                    np.random.shuffle(dataset.data)
                    dataset.data = dataset.data[:math.ceil(len(dataset.data)*train_config["train_ratio"])]
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=train_config["batch_size"], shuffle=data_type=="train", drop_last=False)
                data_loader = DataLoader(dataset,
                    batch_sampler=sampler,
                    collate_fn=EdgeSeqDataset.batchify,
                    pin_memory=data_type=="train")
            data_loaders[data_type] = data_loader
            logger.info("data (data_type: {:<5s}, len: {}) generated".format(data_type, len(dataset.data)))
            logger.info("data_loader (data_type: {:<5s}, len: {}, batch_size: {}) generated".format(data_type, len(data_loader), train_config["batch_size"]))

    # optimizer and losses
    writer = SummaryWriter(save_model_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"], amsgrad=True)
    optimizer.zero_grad()
    scheduler = None
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #     len(data_loaders["train"]), train_config["epochs"]*len(data_loaders["train"]), min_percent=0.0001)

    best_reg_losses = {"train": INF, "dev": INF, "test": INF}
    best_reg_epochs = {"train": -1, "dev": -1, "test": -1}

    for epoch in range(train_config["epochs"]):
        for data_type, data_loader in data_loaders.items():

            if data_type == "train":
                mean_reg_loss, mean_bp_loss = train(model, optimizer, scheduler, data_type, data_loader, device,
                    train_config, epoch, logger=logger, writer=writer)
                torch.save(model.state_dict(), os.path.join(save_model_dir, 'epoch%d.pt' % (epoch)))
            else:
                mean_reg_loss, mean_bp_loss, evaluate_results = evaluate(model, data_type, data_loader, device,
                    train_config, epoch, logger=logger, writer=writer)
                with open(os.path.join(save_model_dir, '%s%d.json' % (data_type, epoch)), "w") as f:
                    json.dump(evaluate_results, f)
            if mean_reg_loss <= best_reg_losses[data_type]:
                best_reg_losses[data_type] = mean_reg_loss
                best_reg_epochs[data_type] = epoch
                logger.info("data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type, mean_reg_loss, epoch))
    for data_type in data_loaders.keys():
        logger.info("data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type, best_reg_losses[data_type], best_reg_epochs[data_type]))

