import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import copy
import numpy as np
from functools import partial
from dgl.nn.pytorch.conv import RelGraphConv
from basemodel import GraphAdjModel
from utils import map_activation_str_to_layer, split_and_batchify_graph_feats


class RGINLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, regularizer="basis", num_bases=None, act_func="relu", dropout=0.0):
        super(RGINLayer, self).__init__()
        self.rgc_layer = RelGraphConv(
            in_feat=in_feat, out_feat=out_feat, num_rels=num_rels,
            regularizer=regularizer, num_bases=num_bases,
            activation=None, self_loop=True, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.Linear(out_feat, out_feat),
            # nn.BatchNorm1d(out_feat),
            map_activation_str_to_layer(act_func), 
            nn.Linear(out_feat, out_feat),
            map_activation_str_to_layer(act_func))
        self.drop = nn.Dropout(dropout)
        
        # init
        if hasattr(self.rgc_layer, "weight") and self.rgc_layer.weight is not None:
            nn.init.normal_(self.rgc_layer.weight, 0.0, 1/(out_feat)**0.5)
        if hasattr(self.rgc_layer, "w_comp") and self.rgc_layer.w_comp is not None:
            nn.init.normal_(self.rgc_layer.w_comp, 0.0, 1/(out_feat)**0.5)
        if hasattr(self.rgc_layer, "loop_weight") and self.rgc_layer.loop_weight is not None:
            nn.init.normal_(self.rgc_layer.loop_weight, 0.0, 1/(out_feat)**0.5)
        if hasattr(self.rgc_layer, "h_bias") and self.rgc_layer.h_bias is not None:
            nn.init.zeros_(self.rgc_layer.h_bias)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 1/(out_feat)**0.5)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, g, x, etypes):
        g = self.rgc_layer(g, x, etypes, norm=None)
        g = self.mlp(g)
        g = self.drop(g)
        return g


class RGIN(GraphAdjModel):
    def __init__(self, config):
        super(RGIN, self).__init__(config)

        # create networks
        p_emb_dim, g_emb_dim = self.get_emb_dim()
        self.g_net, g_dim = self.create_net(
            name="graph", input_dim=g_emb_dim, hidden_dim=config["rgin_hidden_dim"],
            num_layers=config["rgin_graph_num_layers"], 
            num_rels=self.max_ngel, num_bases=config["rgin_num_bases"], regularizer=config["rgin_regularizer"], 
            act_func=self.act_func, dropout=self.dropout)
        self.p_net, p_dim = (self.g_net, g_dim) if self.share_arch else self.create_net(
            name="pattern", input_dim=p_emb_dim, hidden_dim=config["rgin_hidden_dim"],
            num_layers=config["rgin_pattern_num_layers"], 
            num_rels=self.max_npel, num_bases=config["rgin_num_bases"], regularizer=config["rgin_regularizer"], 
            act_func=self.act_func, dropout=self.dropout)
        
        if self.add_enc:
            p_enc_dim, g_enc_dim = self.get_enc_dim()
            p_dim += p_enc_dim
            g_dim += g_enc_dim
        if self.add_degree:
            p_dim += 1
            g_dim += 1
        self.predict_net = self.create_predict_net(config["predict_net"],
            pattern_dim=p_dim, graph_dim=g_dim, hidden_dim=config["predict_net_hidden_dim"],
            num_heads=config["predict_net_num_heads"], recurrent_steps=config["predict_net_recurrent_steps"], 
            mem_len=config["predict_net_mem_len"], mem_init=config["predict_net_mem_init"])

    def create_net(self, name, input_dim, **kw):
        num_layers = kw.get("num_layers", 1)
        hidden_dim = kw.get("hidden_dim", 64)
        num_rels = kw.get("num_rels", 1)
        num_bases = kw.get("num_bases", 8)
        regularizer = kw.get("regularizer", "basis")
        act_func = kw.get("act_func", "relu")
        dropout = kw.get("dropout", 0.0)

        rgins = nn.ModuleList()
        for i in range(num_layers):
            rgins.add_module("%s_rgi%d" % (name, i), RGINLayer(
                in_feat=hidden_dim if i > 0 else input_dim, out_feat=hidden_dim, num_rels=num_rels,
                regularizer=regularizer, num_bases=num_bases,
                act_func=act_func, dropout=dropout))

        return rgins, hidden_dim

    def increase_input_size(self, config):
        old_p_enc_dim, old_g_enc_dim = self.get_enc_dim()
        old_max_npel, old_max_ngel = self.max_npel, self.max_ngel
        super(RGIN, self).increase_input_size(config)
        new_p_enc_dim, new_g_enc_dim = self.get_enc_dim()
        new_max_npel, new_max_ngel = self.max_npel, self.max_ngel

        # increase networks
        if new_max_ngel != old_max_ngel:
            for g_rgin in self.g_net:
                num_bases = g_rgin.rgc_layer.num_bases
                device = g_rgin.rgc_layer.weight.device
                regularizer = g_rgin.rgc_layer.regularizer
                if regularizer == "basis":
                    if num_bases < old_max_ngel:
                        new_w_comp = nn.Parameter(
                            torch.zeros((new_max_ngel, num_bases), dtype=torch.float32, device=device, requires_grad=True))
                        with torch.no_grad():
                            new_w_comp[:old_max_ngel].data.copy_(g_rgin.rgc_layer.w_comp)
                    else:
                        new_w_comp = nn.Parameter(
                            torch.zeros((new_max_ngel, num_bases), dtype=torch.float32, device=device, requires_grad=True))
                        with torch.no_grad():
                            ind = np.diag_indices(num_bases)
                            new_w_comp[ind[0], ind[1]] = 1.0
                    del g_rgin.rgc_layer.w_comp
                    g_rgin.rgc_layer.w_comp = new_w_comp
                elif regularizer == "bdd":
                    new_weight = nn.Parameter(
                        torch.zeros((new_max_ngel, g_rgin.rgc_layer.weight.size(1)),
                            dtype=torch.float32, device=device, requires_grad=True))
                    with torch.no_grad():
                        new_weight[:old_max_ngel].data.copy_(g_rgin.rgc_layer.weight)
                    del g_rgin.rgc_layer.weight
                    g_rgin.rgc_layer.weight = new_weight
                else:
                    raise NotImplementedError
        if self.share_arch:
            del self.p_net
            self.p_net = self.g_net
        elif new_max_npel != old_max_npel:
            for p_rgin in self.p_net:
                num_bases = p_rgin.rgc_layer.num_bases
                device = p_rgin.rgc_layer.weight.device
                regularizer = p_rgin.rgc_layer.regularizer
                if regularizer == "basis":
                    if num_bases < old_max_npel:
                        new_w_comp = nn.Parameter(
                            torch.zeros((new_max_npel, num_bases), dtype=torch.float32, device=device, requires_grad=True))
                        with torch.no_grad():
                            new_w_comp[:old_max_npel].data.copy_(p_rgin.rgc_layer.w_comp)
                    else:
                        new_w_comp = nn.Parameter(
                            torch.zeros((new_max_npel, num_bases), dtype=torch.float32, device=device, requires_grad=True))
                        with torch.no_grad():
                            ind = np.diap_indices(num_bases)
                            new_w_comp[ind[0], ind[1]] = 1.0
                    del p_rgin.rgc_layer.w_comp
                    p_rgin.rgc_layer.w_comp = new_w_comp
                elif regularizer == "bdd":
                    new_weight = nn.Parameter(
                        torch.zeros((max_npel, p_rgin.rgc_layer.weight.size(1)), dtype=torch.float32, device=device, requires_grad=True))
                    with torch.no_grad():
                        new_weight[:old_max_npel].data.copy_(p_rgin.rgc_layer.weight)
                    del p_rgin.rgc_layer.weight
                    p_rgin.rgc_layer.weight = new_weight
                else:
                    raise NotImplementedError

        # increase predict network
        if self.add_enc and (new_g_enc_dim != old_g_enc_dim or new_p_enc_dim != old_p_enc_dim):
            self.predict_net.increase_input_size(
                self.predict_net.pattern_dim+new_p_enc_dim-old_p_enc_dim,
                self.predict_net.graph_dim+new_g_enc_dim-old_g_enc_dim)

    def increase_net(self, config):
        p_emb_dim, g_emb_dim = self.get_emb_dim()
        g_net, g_dim = self.create_net(
            name="graph", input_dim=g_emb_dim, hidden_dim=config["rgin_hidden_dim"],
            num_layers=config["rgin_graph_num_layers"], 
            num_rels=self.max_ngel, num_bases=config["rgin_num_bases"], regularizer=config["rgin_regularizer"], 
            act_func=self.act_func, dropout=self.dropout)
        assert len(g_net) >= len(self.g_net)
        with torch.no_grad():
            for old_g_rgin, new_g_rgin in zip(self.g_net, g_net):
                new_g_rgin.load_state_dict(old_g_rgin.state_dict())
        del self.g_net
        self.g_net = g_net

        if self.share_arch:
            self.p_net = self.g_net
        else:
            p_net, p_dim = self.create_net(
                name="pattern", input_dim=p_emb_dim, hidden_dim=config["rgin_hidden_dim"],
                num_layers=config["rgin_pattern_num_layers"], 
                num_rels=self.max_npel, num_bases=config["rgin_num_bases"], regularizer=config["rgin_regularizer"], 
                act_func=self.act_func, dropout=self.dropout)
            assert len(p_net) >= len(self.p_net)
            with torch.no_grad():
                for old_p_rgin, new_p_rgin in zip(self.p_net, p_net):
                    new_p_rgin.load_state_dict(old_p_rgin.state_dict())
            del self.p_net
            self.p_net = p_net

    def forward(self, pattern, pattern_len, graph, graph_len):
        bsz = pattern_len.size(0)

        gate = self.get_filter_gate(pattern, pattern_len, graph, graph_len)
        zero_mask = (gate == 0) if gate is not None else None
        pattern_emb, graph_emb = self.get_emb(pattern, pattern_len, graph, graph_len)
        if zero_mask is not None:
            graph_emb.masked_fill_(zero_mask, 0.0)

        pattern_output = pattern_emb
        for p_rgin in self.p_net:
            o = p_rgin(pattern, pattern_output, pattern.edata["label"])
            pattern_output = o + pattern_output
        
        graph_output = graph_emb
        for g_rgin in self.g_net:
            o = g_rgin(graph, graph_output, graph.edata["label"])
            graph_output = o + graph_output
            if zero_mask is not None:
                graph_output.masked_fill_(zero_mask, 0.0)
        
        if self.add_enc and self.add_degree:
            pattern_enc, graph_enc = self.get_enc(pattern, pattern_len, graph, graph_len)
            if zero_mask is not None:
                graph_enc.masked_fill_(zero_mask, 0.0)
            pattern_output = torch.cat([pattern_enc, pattern_output, pattern.ndata["indeg"].unsqueeze(-1)], dim=1)
            graph_output = torch.cat([graph_enc, graph_output, graph.ndata["indeg"].unsqueeze(-1)], dim=1)
        elif self.add_enc:
            pattern_enc, graph_enc = self.get_enc(pattern, pattern_len, graph, graph_len)
            if zero_mask is not None:
                graph_enc.masked_fill_(zero_mask, 0.0)
            pattern_output = torch.cat([pattern_enc, pattern_output], dim=1)
            graph_output = torch.cat([graph_enc, graph_output], dim=1)
        elif self.add_degree:
            pattern_output = torch.cat([pattern_output, pattern.ndata["indeg"].unsqueeze(-1)], dim=1)
            graph_output = torch.cat([graph_output, graph.ndata["indeg"].unsqueeze(-1)], dim=1)
        
        pred = self.predict_net(
            split_and_batchify_graph_feats(pattern_output, pattern_len)[0], pattern_len, 
            split_and_batchify_graph_feats(graph_output, graph_len)[0], graph_len)
        
        return pred
