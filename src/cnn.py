import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from basemodel import EdgeSeqModel
from utils import map_activation_str_to_layer

class CNN(EdgeSeqModel):
    def __init__(self, config):
        super(CNN, self).__init__(config)

        if len(config["cnn_conv_kernel_sizes"]) != len(config["cnn_pool_kernel_sizes"]):
            raise ValueError("Error: the size of cnn_conv_kernel_sizes is not equal to that of cnn_pool_kernel_sizes.")
        if len(config["cnn_conv_strides"]) != len(config["cnn_pool_strides"]):
            raise ValueError("Error: the size of cnn_conv_strides is not equal to that of cnn_pool_strides.")
        if len(config["cnn_conv_kernel_sizes"]) != len(config["cnn_conv_strides"]):
            raise ValueError("Error: the size of cnn_conv_kernel_sizes is not equal to that of cnn_conv_strides.")

        # create networks
        p_emb_dim, g_emb_dim = self.get_emb_dim()
        self.g_net, g_dim = self.create_net(
            name="graph", input_dim=g_emb_dim,
            conv_channels=config["cnn_conv_channels"],
            conv_kernel_sizes=config["cnn_conv_kernel_sizes"], conv_paddings=config["cnn_conv_paddings"],
            conv_strides=config["cnn_conv_strides"],
            pool_kernel_sizes=config["cnn_pool_kernel_sizes"], pool_paddings=config["cnn_pool_paddings"],
            pool_strides=config["cnn_pool_strides"],
            act_func=self.act_func, dropout=self.dropout)
        self.p_net, p_dim = (self.g_net, g_dim) if self.share_arch else self.create_net(
            name="pattern", input_dim=p_emb_dim,
            conv_channels=config["cnn_conv_channels"],
            conv_kernel_sizes=config["cnn_conv_kernel_sizes"], conv_paddings=config["cnn_conv_paddings"],
            conv_strides=config["cnn_conv_strides"],
            pool_kernel_sizes=config["cnn_pool_kernel_sizes"], pool_paddings=config["cnn_pool_paddings"],
            pool_strides=config["cnn_pool_strides"],
            act_func=self.act_func, dropout=self.dropout)
        
        # create predict layers
        self.predict_net = self.create_predict_net(config["predict_net"],
            pattern_dim=p_dim, graph_dim=g_dim, hidden_dim=config["predict_net_hidden_dim"],
            num_heads=config["predict_net_num_heads"], recurrent_steps=config["predict_net_recurrent_steps"], 
            mem_len=config["predict_net_mem_len"], mem_init=config["predict_net_mem_init"])

    def create_net(self, name, input_dim, **kw):
        conv_kernel_sizes = kw.get("conv_kernel_sizes", (1,2,3))
        conv_paddings = kw.get("conv_paddings", (-1,-1,-1))
        conv_channels = kw.get("conv_channels", (64,64,64))
        conv_strides = kw.get("conv_strides", (1,1,1))
        pool_kernel_sizes = kw.get("pool_kernel_sizes", (2,3,4))
        pool_strides = kw.get("pool_strides", (1,1,1))
        pool_paddings = kw.get("pool_paddings", (-1,-1,-1))
        act_func = kw.get("act_func", "relu")
        dropout = kw.get("dropout", 0.0)

        cnns = nn.ModuleList()
        for i, conv_kernel_size in enumerate(conv_kernel_sizes):
            conv_stride = conv_strides[i]
            conv_padding = conv_paddings[i]
            if conv_padding == -1:
                conv_padding = conv_kernel_size//2

            pool_kernel_size = pool_kernel_sizes[i]
            pool_padding = pool_paddings[i]
            pool_stride = pool_strides[i]
            if pool_padding == -1:
                pool_padding = pool_kernel_size//2

            cnn = nn.Sequential(OrderedDict([
                ("conv", nn.Conv1d(conv_channels[i-1] if i > 0 else input_dim, conv_channels[i],
                    kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)),
                ("act", map_activation_str_to_layer(act_func)),
                ("pool", nn.MaxPool1d(
                    kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)),
                # ("norm", nn.BatchNorm1d(conv_channels[i])),
                ("drop", nn.Dropout(dropout))]))
            cnns.add_module("%s_cnn%d" % (name, i), cnn)
            num_features = conv_channels[i]

        # init
        for m in cnns.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=act_func)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        return cnns, num_features
    
    def increase_input_size(self, config):
        super(CNN, self).increase_input_size(config)

    def increase_net(self, config):
        p_emb_dim, g_emb_dim = self.get_emb_dim()
        g_net, g_dim = self.create_net(
            name="graph", input_dim=g_emb_dim,
            conv_channels=config["cnn_conv_channels"],
            conv_kernel_sizes=config["cnn_conv_kernel_sizes"], conv_paddings=config["cnn_conv_paddings"],
            conv_strides=config["cnn_conv_strides"],
            pool_kernel_sizes=config["cnn_pool_kernel_sizes"], pool_paddings=config["cnn_pool_paddings"],
            pool_strides=config["cnn_pool_strides"],
            act_func=self.act_func, dropout=self.dropout)
        assert len(g_net) >= len(self.g_net)
        with torch.no_grad():
            for old_g_cnn, new_g_cnn in zip(self.g_net, g_net):
                new_g_cnn.load_state_dict(old_g_cnn.state_dict())
        del self.g_net
        self.g_net = g_net

        if self.share_arch:
            self.p_net = self.g_net
        else:
            p_net, p_dim = self.create_net(
                name="pattern", input_dim=p_emb_dim,
                conv_channels=config["cnn_conv_channels"],
                conv_kernel_sizes=config["cnn_conv_kernel_sizes"], conv_paddings=config["cnn_conv_paddings"],
                conv_strides=config["cnn_conv_strides"],
                pool_kernel_sizes=config["cnn_pool_kernel_sizes"], pool_paddings=config["cnn_pool_paddings"],
                pool_strides=config["cnn_pool_strides"],
                act_func=self.act_func, dropout=self.dropout)
            assert len(p_net) >= len(self.p_net)
            with torch.no_grad():
                for old_p_cnn, new_p_cnn in zip(self.p_net, p_net):
                    new_p_cnn.load_state_dict(old_p_cnn.state_dict())
            del self.p_net
            self.p_net = p_net

    def forward(self, pattern, pattern_len, graph, graph_len):
        bsz = pattern_len.size(0)
        
        gate = self.get_filter_gate(pattern, pattern_len, graph, graph_len)
        zero_mask = (gate == 0).unsqueeze(-1) if gate is not None else None
        pattern_emb, graph_emb = self.get_emb(pattern, pattern_len, graph, graph_len)
        if zero_mask is not None:
            graph_emb.masked_fill_(zero_mask, 0.0)

        pattern_output = pattern_emb.transpose(1, 2)
        for p_cnn in self.p_net:
            o = p_cnn(pattern_output)
            if o.size() == pattern_output.size():
                pattern_output = o + pattern_output
            else:
                pattern_output = o
        pattern_output = pattern_output.transpose(1, 2)

        graph_output = graph_emb.transpose(1, 2)
        for g_cnn in self.g_net:
            o = g_cnn(graph_output)
            if o.size() == graph_output.size():
                graph_output = o + graph_output
            else:
                graph_output = o
        graph_output = graph_output.transpose(1, 2)

        pred = self.predict_net(pattern_output, pattern_len, graph_output, graph_len)

        return pred
