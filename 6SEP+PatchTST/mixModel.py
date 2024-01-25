__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

stream_num = 4

# from collections import OrderedDict

# Cell
class MaskModel(nn.Module):
    def __init__(self, batch_size,patch_num,  patch_len,threshold=0.5, device=torch.device('cuda:0')):
        super().__init__()
        self.mask = nn.Parameter(torch.empty((512,patch_num, 256), device=device))
        torch.nn.init.xavier_normal_(self.mask)
        #self.mask = nn.Parameter(torch.ones((1, input_shape[1], 1), device=device))
        self.init_mask = torch.ones((1,patch_num, 1), device=device)
        self.indices_to_keep = (torch.arange(patch_num, device=device) % patch_len == 0).unsqueeze(-1).float()
        self.threshold = torch.tensor(threshold, device=device)
    
    def forward(self, x):

        mask = torch.sigmoid(self.mask).to(x.device)

        mask = mask * (1 - self.indices_to_keep) + self.indices_to_keep  
        mask = torch.sigmoid((mask - self.threshold) * 10)  
        z = x * mask

        return z
class PatchTST_backbone(nn.Module):
    def __init__(self,
                 seq_len,
                 patch_len,
                 stride,
                 n_heads,
                 vocab_dim,
                 batch_size,
                 n_layers: int = 1,
                 d_model=256,  # Hidden_dim
                 
                 d_ff: int = 4096,
                 norm: str = 'BatchNorm',
                 attn_dropout: float = 0.0,
                 dropout: float = 0.0,
                 act: str = "gelu",
                 key_padding_mask: bool = 'auto',
                 store_attn=True,
                 res_attention=True,
                 **kwargs):
        super().__init__()

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.vocab_dim = vocab_dim

        patch_num = int((seq_len - patch_len) / stride + 1)

        self.input_map = torch.nn.Embedding(256, self.vocab_dim)
        # Backbone
        self.backbone = TSTiEncoder(patch_num=patch_num, patch_len=patch_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act,res_attention=res_attention,
                                    store_attn=store_attn,
                                    key_padding_mask=key_padding_mask,
                                    **kwargs)
   
        self.U_map = torch.nn.Linear(patch_num,8, bias=True)
        torch.nn.init.kaiming_normal_(self.U_map.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.U_map.bias, 0)
        self.mask =MaskModel(batch_size,patch_num,patch_len)
        self.output_logit_map = torch.nn.Linear(256, 256)
        self.X_map = torch.nn.Linear(8, 128, bias=True)
        self.Z_map = torch.nn.Linear(128,1, bias=True)

    def forward(self, x,inputs_list=None,streamPooling=None,):  # z: [bs x nvars x seq_len]
        x = self.input_map(x)
        #x = self._concat_pos_embs(x, 0)
        x = torch.sigmoid(x)


        x = x.unfold(dimension=-2, size=self.patch_len, step=self.stride)
        x = x.permute(0, 1, 3, 2)
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = self.mask(x)

 
        x = x.permute(0, 2, 1)
        
        x = self.U_map(x)
        if inputs_list is not None:
            global stream_num
            for i in range(1,stream_num):
                with torch.cuda.stream(streamPooling[i]):
                    inputs_list[i] = inputs_list[i].to("cuda",non_blocking=True).long()
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.gelu(x)
        skip = x 
        # model
        x = self.backbone(x) 
        # x = (skip +x) /2
        # x = self.output_logit_map(x)
        x = x.transpose(1,2)
        x = self.X_map(x)
        x = torch.nn.functional.gelu(x)
        x = self.Z_map(x)
        x = x.transpose(1,2)

        return x

    def full_loss(self,
                  inputs,inputs_list=None,streamPooling=None,
                  with_grad=True):

        logits = self.forward(inputs[:, :-1],inputs_list=inputs_list,streamPooling=streamPooling)
        logits = logits.transpose(1, 2)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :, -1], inputs[:, -1], reduction='mean')

        if with_grad:
            loss.backward()

        return loss, logits

    def _concat_pos_embs(self, x, start_index):

        pos_emb_size = self.vocab_dim // 2

        positions = torch.arange(
            start_index, start_index + x.shape[1], dtype=x.dtype, device=x.device)
        freqs = torch.exp(
            torch.arange(0, pos_emb_size, 2, dtype=x.dtype, device=x.device) *
            (-np.log(10000) / pos_emb_size))
        args = positions[None, :, None] * freqs[None, None, :]
        sin_pos_embs = torch.sin(args) * torch.ones_like(x[:, :1, :1])
        cos_pos_embs = torch.cos(args) * torch.ones_like(x[:, :1, :1])
        return torch.cat([x, sin_pos_embs, cos_pos_embs], 2)



class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, patch_num, patch_len,
                 n_layers, d_model, n_heads,d_ff,
                 norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=True,
                res_attention=False, pre_norm=False,
                 pe='sincos', learn_pe=True, verbose=False, **kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        
        self.W_P = nn.Linear(64, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len


        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads,d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn)

    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        z = self.encoder(x)  # z: [bs , nvars , patch_num x d_model]

        return z

    # Cell


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                             attn_dropout=attn_dropout, dropout=dropout,
                             activation=activation, res_attention=res_attention,
                             pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask,
                                                         attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=True,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                             proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
                                                attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:  # false
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0.,
                 qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                   res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                         2)  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

"""     池化层32->14


# 划分为四个 (512, 8) 的 tensor
        # x1, x2, x3, x4 = x.split(8, dim=1)

# # 重塑为 (512, 2, 4) 的 tensor，以进行最大池化
#         batch_size = x1.shape[0]
#         x1 = x1.float().view(batch_size, 2, 4)
#         x2 = x2.float().view(batch_size, 2, 4)
#         x3 = x3.float().view(batch_size, 2, 4)

# # 定义一个最大池化层，将 (2, 4) 的数据降为 (2, 1)
#         max_pool = MaxPool2d((1, 4))

# # 对前三个张量进行最大池化，并将其重新塑形为 (512, 2)
#         x1 = max_pool(x1).view(batch_size, 2)
#         x2 = max_pool(x2).view(batch_size, 2)
#         x3 = max_pool(x3).view(batch_size, 2)
        #output = torch.cat((x1, x2, x3, x4), dim=1)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # output = output.to(device)


"""



""" 

        只取后面四分之一
        tensor1, tensor2 = x[:, :72], x[:, 24:]
        result = tensor2
        result = result.long()

"""



"""     
    前3/4 每四个平均为1个

# 将 input 分割为两个张量
        bs,seq_len = x.shape
        tensor1, tensor2 = x[:, :96], x[:, 24:]
        _ , len1 = tensor.shape
        tensor1 = tensor1.float()
        tensor1 = tensor1.view(bs,len1 // 4, 4).mean(dim=2)
        result = torch.cat((tensor1, tensor2), dim=1)
        result = result.long()


"""
