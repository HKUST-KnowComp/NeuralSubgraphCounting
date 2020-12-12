import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from utils import segment_length
from basemodel import EdgeSeqModel
from utils import map_activation_str_to_layer, batch_convert_len_to_mask


class RNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_type, bidirectional, dropout):
        super(RNNLayer, self).__init__()
        if rnn_type == "GRU":
            rnn_layer = nn.GRU
        elif rnn_type == "LSTM":
            rnn_layer = nn.LSTM
        else:
            raise NotImplementedError("Currently, %s is not supported!" % (rnn_type))
        self.rnn = rnn_layer(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)

        # init
        for layer_weights in self.rnn._all_weights:
            for w in layer_weights:
                if "weight" in w:
                    weight = getattr(self.rnn, w)
                    nn.init.orthogonal_(weight)
                elif "bias" in w:
                    bias = getattr(self.rnn, w)
                    if bias is not None:
                        nn.init.zeros_(bias)

    def forward(self, x):
        x, h = self.rnn(x)
        x = self.drop(x)
        return x

class RNN(EdgeSeqModel):
    def __init__(self, config):
        super(RNN, self).__init__(config)

        # create networks
        p_emb_dim, g_emb_dim = self.get_emb_dim()
        self.g_net, g_dim = self.create_net(
            name="graph", input_dim=g_emb_dim, hidden_dim=config["rnn_hidden_dim"],
            num_layers=config["rnn_graph_num_layers"], 
            rnn_type=config["rnn_type"], bidirectional=config["rnn_bidirectional"],
            dropout=self.dropout)
        self.p_net, p_dim = (self.g_net, g_dim) if self.share_arch else self.create_net(
            name="pattern", input_dim=p_emb_dim, hidden_dim=config["rnn_hidden_dim"],
            num_layers=config["rnn_pattern_num_layers"], 
            rnn_type=config["rnn_type"], bidirectional=config["rnn_bidirectional"],
            dropout=self.dropout)
        
        # create predict layers
        if self.add_enc:
            p_enc_dim, g_enc_dim = self.get_enc_dim()
            p_dim += p_enc_dim
            g_dim += g_enc_dim
        self.predict_net = self.create_predict_net(config["predict_net"],
            pattern_dim=p_dim, graph_dim=g_dim, hidden_dim=config["predict_net_hidden_dim"],
            num_heads=config["predict_net_num_heads"], recurrent_steps=config["predict_net_recurrent_steps"], 
            mem_len=config["predict_net_mem_len"], mem_init=config["predict_net_mem_init"])

    def create_net(self, name, input_dim, **kw):
        num_layers = kw.get("num_layers", 3)
        hidden_dim = kw.get("hidden_dim", 64)
        rnn_type = kw.get("rnn_type", "LSTM")
        bidirectional = kw.get("bidirectional", "False")
        dropout = kw.get("dropout", 0.0)

        num_features = hidden_dim*2 if bidirectional else hidden_dim
        rnns = nn.ModuleList()
        for i in range(num_layers):
            rnns.add_module("%s_rnn%d" % (name, i),RNNLayer(
                input_dim=input_dim if i == 0 else num_features, hidden_dim=hidden_dim,
                rnn_type=rnn_type, bidirectional=bidirectional,
                dropout=dropout))

        return rnns, num_features

    def increase_input_size(self, config):
        old_p_enc_dim, old_g_enc_dim = self.get_enc_dim()
        super(RNN, self).increase_input_size(config)
        new_p_enc_dim, new_g_enc_dim = self.get_enc_dim()
        
        # increase predict network
        if self.add_enc and (new_g_enc_dim != old_g_enc_dim or new_p_enc_dim != old_p_enc_dim):
            self.predict_net.increase_input_size(
                self.predict_net.pattern_dim+new_p_enc_dim-old_p_enc_dim,
                self.predict_net.graph_dim+new_g_enc_dim-old_g_enc_dim)

    def increase_net(self, config):
        p_emb_dim, g_emb_dim = self.get_emb_dim()
        g_net, g_dim = self.create_net(
            name="graph", input_dim=g_emb_dim, hidden_dim=config["rnn_hidden_dim"],
            num_layers=config["rnn_graph_num_layers"], 
            rnn_type=config["rnn_type"], bidirectional=config["rnn_bidirectional"],
            dropout=self.dropout)
        assert len(g_net) >= len(self.g_net)
        with torch.no_grad():
            for old_g_rnn, new_g_rnn in zip(self.g_net, g_net):
                new_g_rnn.load_state_dict(old_g_rnn.state_dict())
        del self.g_net
        self.g_net = g_net

        if self.share_arch:
            self.p_net = self.g_net
        else:
            p_net, p_dim = self.create_net(
                name="pattern", input_dim=p_emb_dim, hidden_dim=config["rnn_hidden_dim"],
                num_layers=config["rnn_graph_num_layers"], 
                rnn_type=config["rnn_type"], bidirectional=config["rnn_bidirectional"],
                dropout=self.dropout)
            assert len(p_net) >= len(self.p_net)
            with torch.no_grad():
                for old_p_rnn, new_p_rnn in zip(self.p_net, p_net):
                    new_p_rnn.load_state_dict(old_p_rnn.state_dict())
            del self.p_net
            self.p_net = p_net
    
    def forward(self, pattern, pattern_len, graph, graph_len):
        bsz = pattern_len.size(0)

        gate = self.get_filter_gate(pattern, pattern_len, graph, graph_len)
        zero_mask = (gate == 0).unsqueeze(-1) if gate is not None else None
        pattern_emb, graph_emb = self.get_emb(pattern, pattern_len, graph, graph_len)
        if zero_mask is not None:
            graph_emb.masked_fill_(zero_mask, 0.0)

        pattern_output = pattern_emb
        for p_rnn in self.p_net:
            o = p_rnn(pattern_output)
            pattern_output = o + pattern_output
        pattern_mask = (batch_convert_len_to_mask(pattern_len)==0).unsqueeze(-1)
        pattern_output.masked_fill_(pattern_mask, 0.0)

        graph_output = graph_emb
        for g_rnn in self.g_net:
            o = g_rnn(graph_output)
            graph_output = o + graph_output
            if zero_mask is not None:
                graph_output.masked_fill_(zero_mask, 0.0)
        graph_mask = (batch_convert_len_to_mask(graph_len)==0).unsqueeze(-1)
        graph_output.masked_fill_(graph_mask, 0.0)
        
        if self.add_enc:
            pattern_enc, graph_enc = self.get_enc(pattern, pattern_len, graph, graph_len)
            if zero_mask is not None:
                graph_enc.masked_fill_(zero_mask, 0.0)
            pattern_output = torch.cat([pattern_enc, pattern_output], dim=2)
            graph_output = torch.cat([graph_enc, graph_output], dim=2)
        
        pred = self.predict_net(pattern_output, pattern_len, graph_output, graph_len)
        
        return pred
