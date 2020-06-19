import torch
import os
import numpy as np
import dgl
import logging
import datetime
import math
import sys
import gc
import re
import subprocess
import json
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
from utils import anneal_fn, get_enc_len, load_data, get_best_epochs, get_linear_schedule_with_warmup
from mlp import MLP
from rnn import RNN
from transformerxl import TXL
from cnn import CNN
from resnet import ResNet
from rgcn import RGCN
from rgin import RGIN
from rsin import RSIN
from train import train, evaluate

warnings.filterwarnings("ignore")
INF = float("inf")

finetune_config = {
    "max_npv": 8, # max_number_pattern_vertices: 8, 16, 32
    "max_npe": 8, # max_number_pattern_edges: 8, 16, 32
    "max_npvl": 8, # max_number_pattern_vertex_labels: 8, 16, 32
    "max_npel": 8, # max_number_pattern_edge_labels: 8, 16, 32

    "max_ngv": 64, # max_number_graph_vertices: 64, 512,4096
    "max_nge": 256, # max_number_graph_edges: 256, 2048, 16384
    "max_ngvl": 16, # max_number_graph_vertex_labels: 16, 64, 256
    "max_ngel": 16, # max_number_graph_edge_labels: 16, 64, 256
    
    # "base": 2,

    "gpu_id": -1,
    "num_workers": 12,
    
    "epochs": 100,
    "batch_size": 64,
    "update_every": 1, # actual batch_sizer = batch_size * update_every
    "print_every": 100,
    "share_emb": True, # sharing embedding requires the same vector length
    "share_arch": True, # sharing architectures
    "dropout": 0.2,
    "dropatt": 0.2,

    "predict_net": "SumPredictNet", # MeanPredictNet, SumPredictNet, MaxPredictNet,
                                    # MeanAttnPredictNet, SumAttnPredictNet, MaxAttnPredictNet,
                                    # MeanMemAttnPredictNet, SumMemAttnPredictNet, MaxMemAttnPredictNet,
                                    # DIAMNet
    "predict_net_hidden_dim": 128,
    "predict_net_num_heads": 4,
    "predict_net_mem_len": 4,
    "predict_net_mem_init": "mean", # mean, sum, max, attn, circular_mean, circular_sum, circular_max, circular_attn, lstm
    "predict_net_recurrent_steps": 3,
    
    "reg_loss": "MSE", # MAE, MSE, SMAE
    "bp_loss": "MSE", # MAE, MSE, SMAE
    "bp_loss_slp": "anneal_cosine$1.0$0.01",    # 0, 0.01, logistic$1.0$0.01, linear$1.0$0.01, cosine$1.0$0.01, 
                                                # cyclical_logistic$1.0$0.01, cyclical_linear$1.0$0.01, cyclical_cosine$1.0$0.01
                                                # anneal_logistic$1.0$0.01, anneal_linear$1.0$0.01, anneal_cosine$1.0$0.01
    "lr": 0.001,
    "weight_decay": 0.00001,
    "max_grad_norm": 8,
    
    "train_ratio": 1.0,
    "pattern_dir": "../data/MUTAG/patterns",
    "graph_dir": "../data/MUTAG/raw",
    "metadata_dir": "../data/MUTAG/metadata",
    "save_data_dir": "../data/MUTAG",
    "save_model_dir": "../dumps/MUTAG",
    "load_model_dir": "../dumps/small/RGCN_DIAMNet"
}

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    for i in range(1, len(sys.argv), 2):
        arg = sys.argv[i]
        value = sys.argv[i+1]
        
        if arg.startswith("--"):
            arg = arg[2:]
        if arg not in finetune_config:
            print("Warning: %s is not surported now." % (arg))
            continue
        finetune_config[arg] = value
        try:
            value = eval(value)
            if isinstance(value, (int, float)):
                finetune_config[arg] = value
        except:
            pass
        

    # load config
    if os.path.exists(os.path.join(finetune_config["load_model_dir"], "train_config.json")):
        with open(os.path.join(finetune_config["load_model_dir"], "train_config.json"), "r") as f:
            train_config = json.load(f)
    elif os.path.exists(os.path.join(finetune_config["load_model_dir"], "finetune_config.json")):
        with open(os.path.join(finetune_config["load_model_dir"], "finetune_config.json"), "r") as f:
            train_config = json.load(f)
    else:
        raise FileNotFoundError("finetune_config.json and train_config.json cannot be found in %s" % (os.path.join(finetune_config["load_model_dir"])))
    
    for key in train_config:
        if key not in finetune_config:
            finetune_config[key] = train_config[key]

    ts = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_name = "%s_%s_%s" % (finetune_config["model"], finetune_config["predict_net"], ts)
    save_model_dir = finetune_config["save_model_dir"]
    os.makedirs(save_model_dir, exist_ok=True)

    # save config
    with open(os.path.join(save_model_dir, "finetune_config.json"), "w") as f:
        json.dump(finetune_config, f)

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(os.path.join(save_model_dir, "finetune_log.txt"), 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    # set device
    device = torch.device("cuda:%d" % finetune_config["gpu_id"] if finetune_config["gpu_id"] != -1 else "cpu")
    if finetune_config["gpu_id"] != -1:
        torch.cuda.set_device(device)

    # check model
    if finetune_config["model"] not in ["MLP","CNN", "DN", "RN", "RNN", "TXL", "RGCN", "RGIN", "RGIN"]:
        raise NotImplementedError("Currently, the %s model is not supported" % (finetune_config["model"]))

    # reset the pattern parameters
    if finetune_config["share_emb"]:
        finetune_config["max_npv"], finetune_config["max_npvl"], finetune_config["max_npe"], finetune_config["max_npel"] = \
            finetune_config["max_ngv"], finetune_config["max_ngvl"], finetune_config["max_nge"], finetune_config["max_ngel"]

    # get the best epoch
    if os.path.exists(os.path.join(finetune_config["load_model_dir"], "finetune_log.txt")):
        best_epochs = get_best_epochs(os.path.join(finetune_config["load_model_dir"], "finetune_log.txt"))
    elif os.path.exists(os.path.join(finetune_config["load_model_dir"], "train_log.txt")):
        best_epochs = get_best_epochs(os.path.join(finetune_config["load_model_dir"], "train_log.txt"))
    else:
        raise FileNotFoundError("finetune_log.txt and train_log.txt cannot be found in %s" % (os.path.join(finetune_config["load_model_dir"])))
    logger.info("retrieve the best epoch for training set ({:0>3d}), dev set ({:0>3d}), and test set ({:0>3d})".format(
        best_epochs["train"], best_epochs["dev"], best_epochs["test"]))

    # load the model
    for key in ["dropout", "dropatt"]:
        train_config[key] = finetune_config[key]
    
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

    model.load_state_dict(torch.load(
        os.path.join(finetune_config["load_model_dir"], "epoch%d.pt" % (best_epochs["dev"])), map_location=torch.device("cpu")))
    model.increase_net(finetune_config)
    if not all([train_config[key] == finetune_config[key] for key in [
        "max_npv", "max_npe", "max_npvl", "max_npel", "max_ngv", "max_nge", "max_ngvl", "max_ngel", "share_emb", "share_arch"]]):
        model.increase_input_size(finetune_config)
    if not all([train_config[key] == finetune_config[key] for key in [
        "predict_net", "predict_net_hidden_dim",
        "predict_net_num_heads", "predict_net_mem_len", "predict_net_mem_init", "predict_net_recurrent_steps"]]):
        new_predict_net = model.create_predict_net(finetune_config["predict_net"],
            pattern_dim=model.predict_net.pattern_dim, graph_dim=model.predict_net.graph_dim,
            hidden_dim=finetune_config["predict_net_hidden_dim"],
            num_heads=finetune_config["predict_net_num_heads"], recurrent_steps=finetune_config["predict_net_recurrent_steps"], 
            mem_len=finetune_config["predict_net_mem_len"], mem_init=finetune_config["predict_net_mem_init"])
        del model.predict_net
        model.predict_net = new_predict_net
    model = model.to(device)
    torch.cuda.empty_cache()
    logger.info("load the model based on the dev set (epoch: {:0>3d})".format(best_epochs["dev"]))
    logger.info(model)
    logger.info("num of parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # load data
    os.makedirs(finetune_config["save_data_dir"], exist_ok=True)
    data_loaders = OrderedDict({"train": None, "dev": None, "test": None})
    if all([os.path.exists(os.path.join(finetune_config["save_data_dir"],
        "%s_%s_dataset.pt" % (
            data_type, "dgl" if finetune_config["model"] in ["RGCN", "RGIN", "RGIN"] else "edgeseq"))) for data_type in data_loaders]):

        logger.info("loading data from pt...")
        for data_type in data_loaders:
            if finetune_config["model"] in ["RGCN", "RGIN", "RGIN"]:
                dataset = GraphAdjDataset(list())
                dataset.load(os.path.join(finetune_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                if data_type == "train":
                    np.random.shuffle(dataset.data)
                    dataset.data = dataset.data[:math.ceil(len(dataset.data)*finetune_config["train_ratio"])]
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=finetune_config["batch_size"], shuffle=data_type=="train", drop_last=False)
                data_loader = DataLoader(dataset,
                    batch_sampler=sampler,
                    collate_fn=GraphAdjDataset.batchify,
                    pin_memory=data_type=="train")
            else:
                dataset = EdgeSeqDataset(list())
                dataset.load(os.path.join(finetune_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type)))
                if data_type == "train":
                    np.random.shuffle(dataset.data)
                    dataset.data = dataset.data[:math.ceil(len(dataset.data)*finetune_config["train_ratio"])]
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=finetune_config["batch_size"], shuffle=data_type=="train", drop_last=False)
                data_loader = DataLoader(dataset,
                    batch_sampler=sampler,
                    collate_fn=EdgeSeqDataset.batchify,
                    pin_memory=data_type=="train")
            data_loaders[data_type] = data_loader
            logger.info("data (data_type: {:<5s}, len: {}) generated".format(data_type, len(dataset.data)))
            logger.info("data_loader (data_type: {:<5s}, len: {}, batch_size: {}) generated".format(data_type, len(data_loader), finetune_config["batch_size"]))
    else:
        data = load_data(finetune_config["graph_dir"], finetune_config["pattern_dir"], finetune_config["metadata_dir"], num_workers=finetune_config["num_workers"])
        logger.info("{}/{}/{} data loaded".format(len(data["train"]), len(data["dev"]), len(data["test"])))
        for data_type, x in data.items():
            if finetune_config["model"] in ["RGCN", "RGIN", "RGIN"]:
                if os.path.exists(os.path.join(finetune_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type))):
                    dataset = GraphAdjDataset(list())
                    dataset.load(os.path.join(finetune_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                else:
                    dataset = GraphAdjDataset(x)
                    dataset.save(os.path.join(finetune_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                if data_type == "train":
                    np.random.shuffle(dataset.data)
                    dataset.data = dataset.data[:math.ceil(len(dataset.data)*finetune_config["train_ratio"])]
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=finetune_config["batch_size"], shuffle=data_type=="train", drop_last=False)
                data_loader = DataLoader(dataset,
                    batch_sampler=sampler,
                    collate_fn=GraphAdjDataset.batchify,
                    pin_memory=data_type=="train")
            else:
                if os.path.exists(os.path.join(finetune_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type))):
                    dataset = EdgeSeqDataset(list())
                    dataset.load(os.path.join(finetune_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type)))
                else:
                    dataset = EdgeSeqDataset(x)
                    dataset.save(os.path.join(finetune_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type)))
                if data_type == "train":
                    np.random.shuffle(dataset.data)
                    dataset.data = dataset.data[:math.ceil(len(dataset.data)*finetune_config["train_ratio"])]
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=finetune_config["batch_size"], shuffle=data_type=="train", drop_last=False)
                data_loader = DataLoader(dataset,
                    batch_sampler=sampler,
                    collate_fn=EdgeSeqDataset.batchify,
                    pin_memory=data_type=="train")
            data_loaders[data_type] = data_loader
            logger.info("data (data_type: {:<5s}, len: {}) generated".format(data_type, len(dataset.data)))
            logger.info("data_loader (data_type: {:<5s}, len: {}, batch_size: {}) generated".format(data_type, len(data_loader), finetune_config["batch_size"]))

    # optimizer and losses
    writer = SummaryWriter(save_model_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_config["lr"], weight_decay=finetune_config["weight_decay"], amsgrad=True)
    optimizer.zero_grad()
    scheduler = get_linear_schedule_with_warmup(optimizer,
        len(data_loaders["train"]), train_config["epochs"]*len(data_loaders["train"]), min_percent=0.0001)

    best_reg_losses = {"train": INF, "dev": INF, "test": INF}
    best_reg_epochs = {"train": -1, "dev": -1, "test": -1}

    for epoch in range(finetune_config["epochs"]):
        for data_type, data_loader in data_loaders.items():

            if data_type == "train":
                mean_reg_loss, mean_bp_loss = train(model, optimizer, scheduler, data_type, data_loader, device,
                    finetune_config, epoch, logger=logger, writer=writer)
                torch.save(model.state_dict(), os.path.join(save_model_dir, 'epoch%d.pt' % (epoch)))
            else:
                mean_reg_loss, mean_bp_loss, evaluate_results = evaluate(model, data_type, data_loader, device,
                    finetune_config, epoch, logger=logger, writer=writer)
                with open(os.path.join(save_model_dir, '%s%d.json' % (data_type, epoch)), "w") as f:
                    json.dump(evaluate_results, f)

            if mean_reg_loss <= best_reg_losses[data_type]:
                best_reg_losses[data_type] = mean_reg_loss
                best_reg_epochs[data_type] = epoch
                logger.info("data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type, mean_reg_loss, epoch))
    for data_type in data_loaders.keys():
        logger.info("data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type, best_reg_losses[data_type], best_reg_epochs[data_type]))
