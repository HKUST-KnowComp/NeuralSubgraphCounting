import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import segment_data, segment_length, map_activation_str_to_layer, batch_convert_len_to_mask
from basemodel import EdgeSeqModel

_INF = -1e30

class PositionalEmbedding(nn.Module):
    def __init__(self, d_emb):
        super(PositionalEmbedding, self).__init__()

        self.d_emb = d_emb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_emb, 2.0) / d_emb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb.unsqueeze(0).expand(bsz, -1, -1)
        else:
            return pos_emb.unsqueeze(0)


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, act_func="relu", pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            map_activation_str_to_layer(act_func),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout))

        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

        # init
        for m in self.CoreNet.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 1/(d_model**0.5))
                nn.init.zeros_(m.bias)

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization
            inp = self.layer_norm(inp)

        core_out = self.CoreNet(inp)

        ##### residual connection
        output = core_out + inp
        
        if not self.pre_lnorm:
            ##### layer normalization
            output = self.layer_norm(output)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.k_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.v_net = nn.Linear(d_model, n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        # init
        for m in [self.q_net, self.k_net, self.v_net, self.o_net]:
            nn.init.normal_(m.weight, 0.0, self.scale)

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [bsz x hlen x n_head x d_head]
        bsz, qlen = h.size(0), h.size(1)

        if mems is not None:
            c = torch.cat([mems, h], dim=1)
        else:
            c = h
        klen = c.size(1)

        if self.pre_lnorm:
            ##### layer normalization
            h = self.layer_norm(h)
            c = self.layer_norm(c)

        head_q = self.q_net(h).view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = self.k_net(c).view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = self.v_net(c).view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [bsz x qlen x klen x n_head]
        attn_score = torch.einsum("bind,bjnd->bijn", (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2: # [bsz x klen] -> [bsz x qlen x klen x n_head]
                attn_score.masked_fill_((attn_mask == 0).unsqueeze(1).unsqueeze(-1), _INF)
            elif attn_mask.dim() == 3: # [bsz x qlen x klen] -> [bsz x qlen x klen x n_head]
                attn_score.masked_fill_((attn_mask == 0).unsqueeze(-1), _INF)

        # [bsz x qlen x klen x n_head]
        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)

        # klen = vlen
        # [bsz x qlen x klen x n_head] + [bsz x vlen x n_head x d_head] -> [bsz x qlen x n_head x d_head]
        attn_vec = torch.einsum("bijn,bjnd->bind", (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            bsz, qlen, self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        ##### residual connection
        output = h + attn_out
        
        if not self.pre_lnorm:
            ##### layer normalization
            output = self.layer_norm(output)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
        tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False, **kw):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.k_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.v_net = nn.Linear(d_model, n_head * d_head, bias=False)
        # self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        # init
        for m in [self.q_net, self.k_net, self.v_net, self.o_net]:
            nn.init.normal_(m.weight, 0.0, self.scale)

    def _rel_shift(self, x, zero_triu=False):
        # x: bsz x qlen x klen x n_head
        zero_pad = torch.zeros((x.size(0), x.size(1), 1, x.size(3)),
                               device=x.device, dtype=x.dtype, requires_grad=False)   # bsz x qlen x 1 x n_head
        x_padded = torch.cat([x, zero_pad], dim=2) # bsz x qlen x (klen+1) x n_head

        x = x_padded[:,:,1:,:]

        if zero_triu:
            ones = torch.ones((x.size(1), x.size(2)), device=x.device, dtype=x.dtype, requires_grad=False)
            x = x * torch.tril(ones, diagonal=x.size(2) - x.size(1)).unsqueeze(0).unsqueeze(-1)

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kw):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kw)

        self.r_net = nn.Linear(self.d_model, self.n_head*self.d_head, bias=False)

        # init
        nn.init.normal_(self.r_net.weight, 0.0, 1/((self.n_head*self.d_head)**0.5))

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        # r: [bsz, klen, d_model], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_r_bias: [klen, n_head], used for term D
        bsz, qlen = w.size(0), w.size(1)

        if mems is not None:
            c = torch.cat([mems, w], dim=1)
        else:
            c = w
        klen = c.size(1)
            
        if self.pre_lnorm:
            ##### layer normalization
            w = self.layer_norm(w)
            c = self.layer_norm(c)
        
        r_head_k = self.r_net(r)
        w_head_q = self.q_net(w)
        w_head_k = self.k_net(c)
        w_head_v = self.v_net(c)

        r_head_k = r_head_k.view(klen, self.n_head, self.d_head)                # klen x n_head x d_head
        w_head_q = w_head_q.view(bsz, qlen, self.n_head, self.d_head)           # bsz x qlen x n_head x d_head
        w_head_k = w_head_k.view(bsz, klen, self.n_head, self.d_head)           # bsz x klen x n_head x d_head
        w_head_v = w_head_v.view(bsz, klen, self.n_head, self.d_head)           # bsz x klen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # bsz x qlen x n_head x d_head
        AC = torch.einsum("bind,bjnd->bijn", (rw_head_q, w_head_k))             # bsz x qlen x klen x n_head

        rr_head_q = w_head_q + r_r_bias                                         # bsz x qlen x n_head x d_head
        BD = torch.einsum("bind,jnd->bijn", (rr_head_q, r_head_k))              # bsz x qlen x klen x n_head
        BD = self._rel_shift(BD)

        # [bsz x qlen x klen x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                # bsz x klen -> bsz x qlen x klen x n_head
                attn_score = attn_score.masked_fill_((attn_mask == 0).unsqueeze(1).unsqueeze(-1), _INF)
            elif attn_mask.dim() == 3:
                # bsz x qlen x klen -> bsz x qlen x klen x n_head
                attn_score = attn_score.masked_fill_((attn_mask == 0).unsqueeze(-1), _INF)

        # [bsz x qlen x klen x n_head]
        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum("bijn,bjnd->bind", (attn_prob, w_head_v))

        # [bsz x qlen x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            bsz, qlen, self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        
        ##### residual connection
        output = w + attn_out
        
        if not self.pre_lnorm:
            ##### layer normalization
            output = self.layer_norm(output)

        return output


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kw):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kw)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        bsz, qlen = w.size(0), w.size(1)

        if mems is not None:
            c = torch.cat([mems, w], dim=1)
        else:
            c = w
        klen = c.size(1)

        if self.pre_lnorm:
            ##### layer normalization
            w = self.layer_norm(w)
            c = self.layer_norm(c)
        
        w_head_q = self.q_net(w)
        w_head_k = self.k_net(c)
        w_head_v = self.v_net(c)

        w_head_q = w_head_q.view(bsz, qlen, self.n_head, self.d_head)           # bsz x qlen x n_head x d_head
        w_head_k = w_head_k.view(bsz, klen, self.n_head, self.d_head)           # bsz x qlen x n_head x d_head
        w_head_v = w_head_v.view(bsz, klen, self.n_head, self.d_head)           # bsz x qlen x n_head x d_head

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], dim=0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], dim=0)
        elif klen < r_emb.size(0):
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias.unsqueeze(0).unsqueeze(0)               # bsz x qlen x n_head x d_head

        AC = torch.einsum("bind,bjnd->bijn", (rw_head_q, w_head_k))             # bsz x qlen x klen x n_head
        B_ = torch.einsum("bind,jnd->bijn", (w_head_q, r_emb))                  # bsz x qlen x klen x n_head
        D_ = r_bias.unsqueeze(0).unsqueeze(0)                                   # 1 x 1 x klen x n_head
        BD = self._rel_shift(B_ + D_)

        # [bsz x qlen x klen x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                # bsz x klen -> bsz x qlen x klen x n_head
                attn_score = attn_score.masked_fill(
                    attn_mask.unsqueeze(1).unsqueeze(-1), _INF)
            elif attn_mask.dim() == 3:
                # bsz x qlen x klen -> bsz x qlen x klen x n_head
                attn_score = attn_score.masked_fill(
                    attn_mask.unsqueeze(-1), _INF)

        # [bsz x qlen x klen x n_head]
        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum("bijn,bjnd->bind", (attn_prob, w_head_v))

        # [bsz x qlen x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            bsz, qlen, self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        ##### residual connection
        output = w + attn_out
        
        if not self.pre_lnorm:
            ##### layer normalization
            output = self.layer_norm(output)

        return output


class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kw):
        super(TransformerLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kw)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
            act_func=kw.get("act_func", "relu"), pre_lnorm=kw.get("pre_lnorm"))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelLearnableTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kw):
        super(RelLearnableTransformerLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, **kw)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
            act_func=kw.get("act_func", "relu"), pre_lnorm=kw.get("pre_lnorm"))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelPartialLearnableTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kw):
        super(RelPartialLearnableTransformerLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, **kw)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
            act_func=kw.get("act_func", "relu"), pre_lnorm=kw.get("pre_lnorm"))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class TXL(EdgeSeqModel):
    def __init__(self, config):
        super(TXL, self).__init__(config)

        self.drop = nn.Dropout(self.dropout)
        self.tgt_len = config["txl_tgt_len"]
        self.mem_len = config["txl_mem_len"]
        self.ext_len = config["txl_ext_len"]
        self.max_tgt_len = self.tgt_len + self.ext_len + self.mem_len
        self.clamp_len = config["txl_clamp_len"]
        self.same_length = config["txl_same_len"]
        self.attn_type = config["txl_attn_type"]
        self.d_model = config["txl_d_model"]

        # embedding layers
        p_emb_dim, g_emb_dim = self.get_emb_dim()
        self.emb_scale = 1 / (config["txl_d_head"]**0.5)
        self.g_emb_proj = nn.Linear(g_emb_dim, self.d_model)
        self.p_emb_proj = self.g_emb_proj if self.share_emb else nn.Linear(p_emb_dim, self.d_model)
        self.pos_emb = PositionalEmbedding(self.d_model)

        # transformer layers
        self.g_net, g_dim = self.create_net(
            name="graph", input_dim=self.d_model, num_layers=config["txl_graph_num_layers"],
            d_model=self.d_model, d_inner=config["txl_d_inner"],
            n_head=config["txl_n_head"], d_head=config["txl_d_head"],
            tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
            attn_type=self.attn_type, pre_lnorm=config["txl_pre_lnorm"],
            act_func=self.act_func, dropout=self.dropout, dropatt=self.dropout)
        self.p_net, p_dim = (self.g_net, g_dim) if self.share_arch else self.create_net(
            name="pattern", input_dim=self.d_model, num_layers=config["txl_pattern_num_layers"],
            d_model=self.d_model, d_inner=config["txl_d_inner"],
            n_head=config["txl_n_head"], d_head=config["txl_d_head"],
            tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
            attn_type=self.attn_type, pre_lnorm=config["txl_pre_lnorm"],
            act_func=self.act_func, dropout=self.dropout, dropatt=self.dropout)

        self.g_params = self.create_params(
            num_layers=config["txl_graph_num_layers"], attn_type=self.attn_type,
            n_head=config["txl_d_head"], d_head=config["txl_d_head"], max_tgt_len=self.max_tgt_len)
        self.p_params = self.g_params if self.share_arch else self.create_params(
            num_layers=config["txl_pattern_num_layers"], attn_type=self.attn_type,
            n_head=config["txl_d_head"], d_head=config["txl_d_head"], max_tgt_len=self.max_tgt_len)

        # predict layers
        if self.add_enc:
            p_enc_dim, g_enc_dim = self.get_enc_dim()
            p_dim += p_enc_dim
            g_dim += g_enc_dim
        self.predict_net = self.create_predict_net(config["predict_net"],
            pattern_dim=p_dim, graph_dim=g_dim, hidden_dim=config["predict_net_hidden_dim"],
            num_heads=config["predict_net_num_heads"], recurrent_steps=config["predict_net_recurrent_steps"], 
            mem_len=config["predict_net_mem_len"], mem_init=config["predict_net_mem_init"])

        # init
        nn.init.normal_(self.g_emb_proj.weight, 0.0, self.emb_scale)
        nn.init.zeros_(self.g_emb_proj.bias)
        nn.init.normal_(self.p_emb_proj.weight, 0.0, self.emb_scale)
        nn.init.zeros_(self.p_emb_proj.bias)
        
    def create_net(self, name, input_dim, **kw):
        num_layers = kw.get("num_layers", 1)
        d_model = kw.get("d_model", 64)
        n_head = kw.get("n_head", 8)
        d_head = kw.get("d_head", 8)
        d_inner = kw.get("d_inner", 64)
        tgt_len = kw.get("tgt_len", 64)
        ext_len = kw.get("ext_len", 0)
        mem_len = kw.get("mem_len", 64)
        attn_type = kw.get("attn_type", 0)
        pre_lnorm = kw.get("pre_lnorm", True)
        act_func = kw.get("act_func", "relu")
        dropatt = kw.get("dropatt", 0.0)
        dropout = kw.get("dropout", 0.0)

        txl = nn.ModuleList()
        if attn_type == 0: # the default attention
            for i in range(num_layers):
                txl.add_module("%s_txl(%d)%d" % (name, attn_type, i), RelPartialLearnableTransformerLayer(
                    n_head, d_model, d_head, d_inner, dropout, act_func=act_func,
                    tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                    dropatt=dropatt, pre_lnorm=pre_lnorm))
        elif attn_type == 1: # learnable embeddings
            for i in range(num_layers):
                txl.add_module("%s_txl(%d)%d" % (name, attn_type, i), RelLearnableTransformerLayer(
                    n_head, d_model, d_head, d_inner, dropout, act_func=act_func,
                    tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                    dropatt=dropatt, pre_lnorm=pre_lnorm))
        elif attn_type in [2, 3]: # absolute embeddings
            for i in range(num_layers):
                txl.add_module("%s_txl(%d)%d" % (name, attn_type, i), TransformerLayer(
                    n_head, d_model, d_head, d_inner, dropout, act_func=act_func,
                    dropatt=dropatt, pre_lnorm=pre_lnorm))
        num_features = d_model
        return txl, num_features

    def create_params(self, **kw):
        num_layers = kw.get("num_layers", 6)
        attn_type = kw.get("attn_type", 0)
        n_head = kw.get("n_head", 8)
        d_head = kw.get("d_head", 8)
        max_tgt_len = kw.get("max_tgt_len", 128)

        params = nn.ParameterDict()
        if attn_type == 0: # default attention
            params["r_w_bias"] = nn.Parameter(torch.Tensor(n_head, d_head))
            params["r_r_bias"] = nn.Parameter(torch.Tensor(n_head, d_head))
        elif attn_type == 1: # learnable
            params["r_emb"] = nn.Parameter(torch.Tensor(
                    num_layers, max_tgt_len, n_head, d_head))
            params["r_w_bias"] = nn.Parameter(torch.Tensor(
                    num_layers, n_head, d_head))
            params["r_bias"] = nn.Parameter(torch.Tensor(
                    num_layers, max_tgt_len, n_head))
        elif attn_type == 2: # absolute standard
            pass
        elif attn_type == 3: # absolute deeper SA
            params["r_emb"] = nn.Parameter(torch.Tensor(
                    num_layers, max_tgt_len, n_head, d_head))

        # init
        if hasattr(params, "r_emb"):
            nn.init.normal_(params.r_emb, 0.0, 1/(d_head**0.5))
        if hasattr(params, "r_w_bias"):
            nn.init.normal_(params.r_w_bias, 0.0, 1/(d_head**0.5))
        if hasattr(params, "r_r_bias"):
            nn.init.normal_(params.r_r_bias, 0.0, 1/(d_head**0.5))
        if hasattr(params, "r_bias"):
            nn.init.zeros_(params.r_bias)

        return params

    def reset_length(self, tgt_len, ext_len, mem_len):
        # If the model does not use memory at all, make the ext_len longer.
        # Otherwise, make the mem_len longer and keep the ext_len the same.
        self.tgt_len = tgt_len
        self.ext_len = ext_len
        self.mem_len = mem_len
        assert self.max_tgt_len == self.tgt_len + self.ext_len + self.mem_len
        
    def init_mems(self, num_layers, x):
        if self.mem_len > 0:
            mems = []
            for i in range(num_layers+1):
                empty = torch.empty((x.size(0), 0, self.d_model), dtype=x.dtype, device=x.device)
                mems.append(empty)

            return mems
        else:
            return None

    def update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None: 
            return None

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        new_mems = []
        end_idx = mlen + max(0, qlen - 0 - self.ext_len)
        beg_idx = max(0, end_idx - self.mem_len)
        for i in range(len(hids)):
            if mems is None or mlen == 0:
                new_mems.append(hids[i][:,beg_idx:end_idx].detach())
            else:
                cat = torch.cat([mems[i], hids[i]], dim=1)
                new_mems.append(cat[:,beg_idx:end_idx].detach())
        return new_mems

    def _forward(self, x, x_len, txl, params, attn_mask=None, mems=None):
        bsz, qlen = x.size(0), x.size(1)

        mlen = mems[0].size(1) if mems is not None else 0
        klen = mlen + qlen

        hids = []
        if self.attn_type == 0: # default
            pos_seq = torch.arange(klen-1, -1, -1.0, device=x.device, dtype=x.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(x)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(txl):
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, params["r_w_bias"], params["r_r_bias"],
                    dec_attn_mask=attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 1: # learnable
            core_out = self.drop(x)
            hids.append(core_out)
            for i, layer in enumerate(txl):
                if self.clamp_len > 0:
                    r_emb = params["r_emb"][i][-self.clamp_len :]
                    r_bias = params["r_bias"][i][-self.clamp_len :]
                else:
                    r_emb, r_bias = params["r_emb"][i], params["r_bias"][i]
                r_w_bias = params["r_w_bias"][i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, r_w_bias, r_bias,
                    dec_attn_mask=attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2: # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=x.device, dtype=x.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(x + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(txl):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(x)

            hids.append(core_out)
            for i, layer in enumerate(txl):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = params["r_emb"][i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += params["r_emb"][i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=attn_mask, mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self.update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def encoder_forward(self, enc_inp, enc_len, enc_txl, enc_params, mems=None):
        qlen = enc_inp.size(1)
        mlen = mems[0].size(1) if mems is not None else 0
        enc_attn_mask = batch_convert_len_to_mask(enc_len + mlen, max_seq_len=qlen+mlen)

        return self._forward(enc_inp, enc_len, enc_txl, enc_params, attn_mask=enc_attn_mask, mems=mems)
    
    def decoder_forward(self, dec_inp, dec_len, dec_txl, dec_params, mems=None):
        bsz, qlen = dec_inp.size(0), dec_inp.size(1)
        mlen = mems[0].size(1) if mems is not None else 0
        klen = mlen + qlen
        
        ones = torch.ones((qlen, klen), dtype=torch.uint8, device=dec_inp.device, requires_grad=False)
        if self.same_length:
            mask_len = klen - self.tgt_mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (1 - (torch.triu(ones, diagonal=1+mlen) + torch.tril(ones, -mask_shift_len))).unsqueeze(0)
        else:
            dec_attn_mask = (1 - torch.triu(ones, diagonal=1+mlen)).unsqueeze(0)
        return self._forward(dec_inp, dec_len, dec_txl, dec_params, attn_mask=dec_attn_mask, mems=mems)

    def increase_input_size(self, config):
        old_p_enc_dim, old_g_enc_dim = self.get_enc_dim()
        super(TXL, self).increase_input_size(config)
        new_p_enc_dim, new_g_enc_dim = self.get_enc_dim()

        # increase predict network
        if self.add_enc and (new_g_enc_dim != old_g_enc_dim or new_p_enc_dim != old_p_enc_dim):
            self.predict_net.increase_input_size(
                self.predict_net.pattern_dim+new_p_enc_dim-old_p_enc_dim,
                self.predict_net.graph_dim+new_g_enc_dim-old_g_enc_dim)

    def increase_net(self, config):
        p_emb_dim, g_emb_dim = self.get_emb_dim()
        g_net, g_dim = self.create_net(
            name="graph", input_dim=self.d_model, num_layers=config["txl_graph_num_layers"],
            d_model=self.d_model, d_inner=config["txl_d_inner"],
            n_head=config["txl_n_head"], d_head=config["txl_d_head"],
            tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
            attn_type=self.attn_type, pre_lnorm=config["txl_pre_lnorm"],
            act_func=self.act_func, dropout=self.dropout, dropatt=self.dropout)
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
                name="pattern", input_dim=self.d_model, num_layers=config["txl_pattern_num_layers"],
                d_model=self.d_model, d_inner=config["txl_d_inner"],
                n_head=config["txl_n_head"], d_head=config["txl_d_head"],
                tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                attn_type=self.attn_type, pre_lnorm=config["txl_pre_lnorm"],
                act_func=self.act_func, dropout=self.dropout, dropatt=self.dropout)
            assert len(p_net) >= len(self.p_net)
            with torch.no_grad():
                for old_p_rnn, new_p_rnn in zip(self.p_net, p_net):
                    new_p_rnn.load_state_dict(old_p_rnn.state_dict())
            del self.p_net
            self.p_net = p_net

        g_params = self.create_params(
            num_layers=config["txl_graph_num_layers"], attn_type=self.attn_type,
            n_head=config["txl_d_head"], d_head=config["txl_d_head"], max_tgt_len=self.max_tgt_len)
        with torch.no_grad():
            for k in self.g_params:
                g_params[k].data.copy_(self.g_params[k])
        del self.g_params
        self.g_params = g_params

        if self.share_arch:
            self.p_params = self.g_params
        else:
            p_params = self.g_params if self.share_arch else self.create_params(
                num_layers=config["txl_pattern_num_layers"], attn_type=self.attn_type,
                n_head=config["txl_d_head"], d_head=config["txl_d_head"], max_tgt_len=self.max_tgt_len)
            with torch.no_grad():
                for k in self.p_params:
                    p_params[k].data.copy_(self.p_params[k])
            del self.p_params
            self.p_params = p_params

    def forward(self, pattern, pattern_len, graph, graph_len):
        # data, target, *mems
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        bsz = pattern_len.size(0)

        gate = self.get_filter_gate(pattern, pattern_len, graph, graph_len)
        zero_mask = (gate == 0).unsqueeze(-1) if gate is not None else None
        pattern_emb, graph_emb = self.get_emb(pattern, pattern_len, graph, graph_len)
        if zero_mask is not None:
            graph_emb.masked_fill_(zero_mask, 0.0)

        pattern_emb = self.p_emb_proj(pattern_emb).mul_(self.emb_scale)
        graph_emb = self.g_emb_proj(graph_emb).mul_(self.emb_scale)

        pattern_segments = segment_data(pattern_emb, self.tgt_len)
        pattern_seg_lens = segment_length(pattern_len, self.tgt_len)
        graph_segments = segment_data(graph_emb, self.tgt_len)
        graph_seg_lens = segment_length(graph_len, self.tgt_len)

        pattern_outputs = list()
        for i, (pattern_seg, pattern_seg_len) in enumerate(zip(pattern_segments, pattern_seg_lens)):
            if i == 0:
                pattern_mems = self.init_mems(len(self.p_net), pattern_seg)
            pattern_output, pattern_mems = self.encoder_forward(pattern_seg, pattern_seg_len, self.p_net, self.p_params, mems=pattern_mems)
            pattern_outputs.append(pattern_output)
        pattern_output = torch.cat(pattern_outputs, dim=1)[:,:pattern_emb.size(1)]
        # some segments may only have padded elements, we need to set them as 0 manually
        pattern_mask = (batch_convert_len_to_mask(pattern_len, max_seq_len=pattern_output.size(1))==0).unsqueeze(-1)
        pattern_output.masked_fill_(pattern_mask, 0.0)

        graph_outputs = list()
        for i, (graph_seg, graph_seg_len) in enumerate(zip(graph_segments, graph_seg_lens)):
            if i == 0:
                graph_mems = self.init_mems(len(self.g_net), graph_seg)
            graph_output, graph_mems = self.encoder_forward(graph_seg, graph_seg_len, self.g_net, self.g_params, mems=graph_mems)
            graph_outputs.append(graph_output)
        graph_output = torch.cat(graph_outputs, dim=1)[:,:graph_emb.size(1)]
        # some segments may only have padded elements, we need to set them as 0 manually
        graph_mask = (batch_convert_len_to_mask(graph_len, max_seq_len=graph_output.size(1))==0).unsqueeze(-1)
        graph_output.masked_fill_(graph_mask, 0.0)
        
        if self.add_enc:
            pattern_enc, graph_enc = self.get_enc(pattern, pattern_len, graph, graph_len)
            if zero_mask is not None:
                graph_enc.masked_fill_(zero_mask, 0.0)
            pattern_output = torch.cat([pattern_enc, pattern_output], dim=2)
            graph_output = torch.cat([graph_enc, graph_output], dim=2)
        
        pred = self.predict_net(pattern_output, pattern_len, graph_output, graph_len)
        
        return pred
