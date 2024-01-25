# coding=utf-8

import numpy as np
import torch
import time
import numerator_and_denominator as num_and_den

import torch.nn.functional as F
from torch import nn
def valid_feature_type(feature_type):
    bool1 = feature_type in ['relu', 'elu+1', 'sqr', 'favor+']
    bool2 = feature_type.startswith('favor+') and feature_type.split(
        '_')[1].isdigit()
    return bool1 or bool2
class MaskModel(nn.Module):
    def __init__(self, input_shape=(512, 61, 256), threshold=0.5, patch=4, device=torch.device('cuda:0')):
        super().__init__()
        self.mask = nn.Parameter(torch.empty((512, input_shape[1], 256), device=device))
        torch.nn.init.xavier_normal_(self.mask)
        #self.mask = nn.Parameter(torch.ones((1, input_shape[1], 1), device=device))
        self.init_mask = torch.ones((1, input_shape[1], 1), device=device)
        self.indices_to_keep = (torch.arange(input_shape[1], device=device) % patch == 0).unsqueeze(-1).float()
        self.threshold = torch.tensor(threshold, device=device)
        self.last_z_dim = None  # 用于保存z的第二维度的大小
    def forward(self, x):

        mask = torch.sigmoid(self.mask).to(x.device)

        mask = mask * (1 - self.indices_to_keep) + self.indices_to_keep  
        mask = torch.sigmoid((mask - self.threshold) * 64)  
  
        z = x * mask

        return z

class SLiMPerformer(torch.nn.Module):

    def __init__(self, vocab_size, vocab_dim, hidden_dim, n_layers, ffn_dim, n_heads, feature_type, compute_type):
        super(SLiMPerformer, self).__init__()

        self._vocab_size = vocab_size
        self._vocab_dim = 64
        self._hidden_dim = hidden_dim
        self._scale = hidden_dim // vocab_dim
        self.input_map = torch.nn.Embedding(vocab_size, self._vocab_dim // 2)
        self.output_logit_map = torch.nn.Linear(hidden_dim, vocab_size)

        self.layers = torch.nn.ModuleList([
            SLiMPerformerLayer(hidden_dim, ffn_dim, n_heads, feature_type,
                               compute_type) for _ in range(n_layers)
        ])

        self.mask = MaskModel()
        self.U_map = torch.nn.Linear(61, 4, bias=True)
        torch.nn.init.kaiming_normal_(self.U_map.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.U_map.bias, 0)
        self.X_map = torch.nn.Linear(4, 128, bias=True)
        self.Z_map = torch.nn.Linear(128,1, bias=True)
        
    def forward(self, x,inputs_list=None,streamPooling=None):


        x = self.input_map(x)
        x = self._concat_pos_embs(x, 0)

        x = torch.sigmoid(x)
        x = x.unfold(dimension=-2, size=4, step=1)
        x = x.permute(0, 1, 3, 2)

        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))

        
        x = self.mask(x)
        
        
        x = x.permute(0, 2, 1)
        x = self.U_map(x)
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.gelu(x)
        skip = x
        for layer in self.layers:
            x = layer.full_forward(x, layer.attention.sample_rfs(x.device))
        #前残差
        #x =(skip +x) /2 
        #x = self.output_logit_map(x)
        x = x.transpose(1,2)
        x = self.X_map(x)
        x = torch.nn.functional.gelu(x)  
        #x = x * torch.nn.functional.sigmoid(x)
        x = self.Z_map(x)
        x = x.transpose(1,2)
        # 512,1,256
        
        
#         x =(skip +x) /2 
#         x = self.output_logit_map(x)
        
    
        return x

    def full_loss(self,
                  inputs,inputs_list=None,streamPooling=None,
                  with_grad=True):

        logits = self.forward(inputs[:, :-1],inputs_list,streamPooling)
        logits = logits.transpose(1, 2)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :, -1], inputs[:, -1], reduction='mean')

        if with_grad:
            if inputs_list is not None:
                for i in range(1,4):
                    with torch.cuda.stream(streamPooling[i]):
                        inputs_list[i] = inputs_list[i].to("cuda",non_blocking=True).long()
            loss.backward()

        return loss, logits
    # def full_loss(self, inputs, with_grad=True, lambda_mask=0.001):
    #     logits = self.forward(inputs[:, :-1])
    #     logits = logits.transpose(1, 2)
    #     cross_entropy_loss = torch.nn.functional.cross_entropy(logits[:, :, -1], inputs[:, -1], reduction='mean')
    
    #     mask_loss = torch.norm(self.mask.mask, p=2)  # Compute L1 norm of the mask

    #     total_loss = cross_entropy_loss + lambda_mask * mask_loss  # Add regularization term

    #     if with_grad:
    #         total_loss.backward()
    
    #     return total_loss, logits
    def _concat_pos_embs(self, x, start_index):

        pos_emb_size = self._vocab_dim // 2

        positions = torch.arange(
            start_index, start_index + x.shape[1], dtype=x.dtype, device=x.device)
        freqs = torch.exp(
            torch.arange(0, pos_emb_size, 2, dtype=x.dtype, device=x.device) *
            (-np.log(10000) / pos_emb_size))
        args = positions[None, :, None] * freqs[None, None, :]
        sin_pos_embs = torch.sin(args) * torch.ones_like(x[:, :1, :1])
        cos_pos_embs = torch.cos(args) * torch.ones_like(x[:, :1, :1])
        return torch.cat([x, sin_pos_embs, cos_pos_embs], 2)


class SLiMPerformerLayer(torch.nn.Module):

    def __init__(self, hidden_dim, ffn_dim, n_heads, feature_type, compute_type):
        super(SLiMPerformerLayer, self).__init__()

        self.attention = MultiHeadAttention(feature_type, n_heads, hidden_dim,
                                            compute_type)

        self.U_map = torch.nn.Linear(hidden_dim, ffn_dim)
        self.V_map = torch.nn.Linear(ffn_dim, hidden_dim)
        self.layernorm1 = torch.nn.LayerNorm(hidden_dim)
        self.layernorm2 = torch.nn.LayerNorm(hidden_dim)

    def full_forward(self, x, rfs):
        skip = x

        x = self.layernorm1(x)

        x = self.attention.full_forward(x, rfs)

        x = skip + x

        x = self._ffn(x)
        x = self._ffn(x)

        return x

    def _ffn(self, x):
        skip = x

        x = self.layernorm2(x)

        x = self.U_map(x)
        x = torch.nn.functional.gelu(x)
        x = self.V_map(x)

        x = skip + x

        return x


class MultiHeadAttention(torch.nn.Module):
    """Explicit multihead attention using prefix sum."""

    def __init__(self, feature_type, n_heads, hidden_dim, compute_type):

        super(MultiHeadAttention, self).__init__()

        self._feature_type = feature_type
        self._n_heads = n_heads
        self._hidden_dim = hidden_dim
        self._compute_type = compute_type

        self.q_map = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_map = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_map = torch.nn.Linear(hidden_dim, hidden_dim)

    def full_forward(self, x, rfs):

        queries, keys, values = self._get_queries_keys_values(x, rfs)

        num_sums, den_sums = self.init_sums(x.device)

        if self._compute_type == 'iter':
            num, _ = num_and_den.num_iter(queries, keys, values, num_sums)
            den, _ = num_and_den.den_iter(queries, keys, den_sums)
        elif self._compute_type == 'ps':
            num, _ = num_and_den.num_ps(queries, keys, values, num_sums, False)
            den, _ = num_and_den.den_ps(queries, keys, den_sums, False)
        else:
            num, _ = num_and_den.num_ps(queries, keys, values, num_sums, True)
            den, _ = num_and_den.den_ps(queries, keys, den_sums, True)

        num = torch.transpose(num, 0, 1)
        den = torch.transpose(den, 0, 1)

        outputs = num / (den[Ellipsis, None] + 1e-16)
        outputs = outputs.reshape(x.shape)

        return outputs

    def init_sums(self, device):

        head_dim = self._hidden_dim // self._n_heads

        if self._feature_type.startswith('favor+_'):
            splitted = self._feature_type.split('_')
            feature_dim = int(splitted[1]) * head_dim
        else:
            feature_dim = head_dim

        num_sums = torch.zeros([1, self._n_heads, feature_dim, head_dim],
                               device=device)
        den_sums = torch.zeros([1, self._n_heads, feature_dim], device=device)

        return num_sums, den_sums

    def _get_queries_keys_values(self, inputs, rfs):

        queries = self.q_map(inputs)
        keys = self.k_map(inputs)
        values = self.v_map(inputs)

        queries = queries.reshape(
            [queries.shape[0], queries.shape[1], self._n_heads, -1])
        keys = keys.reshape([keys.shape[0], keys.shape[1], self._n_heads, -1])
        values = values.reshape(
            [values.shape[0], values.shape[1], self._n_heads, -1])

        if self._feature_type == 'relu':
            queries = torch.nn.functional.relu(queries)
            keys = torch.nn.functional.relu(keys)
        elif self._feature_type == 'elu+1':
            queries = torch.nn.functional.elu(queries) + 1
            keys = torch.nn.functional.elu(keys) + 1
        elif self._feature_type == 'sqr':
            queries = queries ** 2
            keys = keys ** 2
        elif self._feature_type == 'abs':
            queries = torch.abs(queries)
            keys = torch.abs(keys)
        else:

            head_dim = self._hidden_dim // self._n_heads

            queries = queries * np.power(head_dim, -0.25)
            queries = torch.einsum('ijkl,klm->ijkm', queries, rfs) - (queries ** 2).sum(
                3, keepdim=True) / 2
            queries = torch.exp(queries)

            keys = keys * np.power(head_dim, -0.25)
            keys = torch.einsum('ijkl,klm->ijkm', keys, rfs) - (keys ** 2).sum(
                3, keepdim=True) / 2
            keys = torch.exp(keys)

        queries = queries.transpose(0, 1)
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)

        return queries, keys, values

    def sample_rfs(self, device):

        if not self._feature_type.startswith('favor+'):
            return None

        if self._feature_type == 'favor+':
            factor = 1
        else:
            splitted = self._feature_type.split('_')
            factor = int(splitted[1])

        head_dim = self._hidden_dim // self._n_heads

        rfs = [[
            _sample_orth_matrix(head_dim, device)[None, Ellipsis] for _ in range(factor)
        ] for _ in range(self._n_heads)]
        rfs = [torch.cat(x, 2) for x in rfs]
        rfs = torch.cat(rfs, 0)
        rfs = rfs * np.sqrt(head_dim)

        return rfs


def _sample_orth_matrix(size, device):
    """Samples orthogonal matrix to reduce variance for random features."""
    subspace = torch.randn(size, size, device=device)
    subspace = torch.tril(subspace)
    subspace = subspace / torch.sqrt((subspace ** 2).sum(0, keepdim=True))

    S = torch.triu(subspace.T.mm(subspace)) - 0.5 * torch.eye(
        subspace.shape[1], device=device)

    result = torch.eye(
        subspace.shape[0], device=device) - subspace.mm(torch.inverse(S)).mm(
        subspace.T)

    return result


class MyConv1dSequence1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(MyConv1dSequence1, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.conv1.bias, 0)

        self.gelu1 = nn.GELU()

    def forward(self, x):
        x = self.gelu1(self.conv1(x))

        return x
    
class LinearLayer(torch.nn.Module):

  def __init__(self, hidden_dim, ffn_dim, out_dim, n_heads, ea=[True, True]):

    super(LinearLayer, self).__init__()

    self.U_map = torch.nn.Linear(hidden_dim, ffn_dim, bias=True)
    torch.nn.init.normal_(self.U_map.weight, 0, 0.01)
    torch.nn.init.normal_(self.U_map.bias, 0, 0.01)
    
    self.V_map = torch.nn.Linear(ffn_dim, out_dim, bias=True)
    torch.nn.init.normal_(self.V_map.weight, 0, 0.01)
    torch.nn.init.normal_(self.V_map.bias, 0, 0.01)
    
    self.layernorm1 = torch.nn.LayerNorm(hidden_dim, eps=1e-05, elementwise_affine=ea[0])
    self.layernorm2 = torch.nn.LayerNorm(out_dim, eps=1e-05, elementwise_affine=ea[1])
    self.ln1 = 1

  def full_forward(self, x):

    x = self._ffn(x)

    return x

  def _ffn(self, x):
    
    if self.ln1:
      x = self.layernorm1(x)
         
    skip = x
    
    x = self.U_map(x)
    x = torch.nn.functional.gelu(x)
    x = self.V_map(x)
    
    x = self.layernorm2(x)
    x = torch.nn.functional.gelu(x)
    
    x = (skip + x)/2


    return x
