# coding=utf-8
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time
import random
import math

inputs_list=None
streamPooling=None
streamcount = 0


class MaskModel(nn.Module):
    def __init__(self, input_shape=(512,33, 256), threshold=0.5, patch=16, device=torch.device('cuda:0')):
        super().__init__()
        self.mask = nn.Parameter(torch.empty((1, input_shape[1], 1), device=device))
        torch.nn.init.xavier_normal_(self.mask)
        #self.mask = nn.Parameter(torch.ones((1, input_shape[1], 1), device=device))
        self.init_mask = torch.ones((1, input_shape[1], 1), device=device)
        self.indices_to_keep = (torch.arange(input_shape[1], device=device) % 8 == 0).unsqueeze(-1).float()
        self.threshold = torch.tensor(threshold, device=device)
        self.last_z_dim = None  # 用于保存z的第二维度的大小
    def forward(self, x):

        mask = torch.sigmoid(self.mask).to(x.device)

        mask = mask * (1 - self.indices_to_keep) + self.indices_to_keep  
        mask = torch.sigmoid((mask - self.threshold) * 10)  
        z = x * mask

        return z

class mlpCompressor(torch.nn.Module):

  def __init__(self, vocab_size, vocab_dim,  hidden_dim, n_layers, ffn_dim, n_heads,
               batch_size):
    super(mlpCompressor, self).__init__()

    self._vocab_size = vocab_size
    self._vocab_dim = vocab_dim
    self._hidden_dim = hidden_dim
    self._scale = hidden_dim // vocab_dim
    self.input_map = torch.nn.Embedding(vocab_size, vocab_dim)
    self.output_logit_map = torch.nn.Linear(self._hidden_dim, vocab_size)
    
    torch.nn.init.normal_(self.input_map.weight, 0, 0.01)

    self.batch_size = batch_size 
    l = []
    
    l.append(BELayer(1, 16, 64, batch_size, [True, True]))
    l.append(LinearLayer(self._hidden_dim, 4096, self._hidden_dim, batch_size, [True, True]))
    l.append(BELayer(1, 32, 64, batch_size, [True, True]))
    l.append(LinearLayer(self._hidden_dim, 4096, self._hidden_dim, batch_size, [True, True]))
    l.append(BELayer(1, 64, 64, batch_size, [True, True]))
    l.append(LinearLayer(self._hidden_dim, 4096, self._hidden_dim, batch_size, [True, True]))
    l.append(BELayer(1, 128, 64, batch_size, [True, True]))
    l.append(LinearLayer(self._hidden_dim, 4096, self._hidden_dim, batch_size, [True, True]))
    l.append(BELayer(1, 256, 64, batch_size, [True, True]))
    l.append(LinearLayer(self._hidden_dim, 4096, self._hidden_dim, batch_size, [True, True]))
    # self.X_map = torch.nn.Linear(256, 4096, bias=True)
    # self.Z_map = torch.nn.Linear(4096,256, bias=True)
    
    self.layers = torch.nn.ModuleList(l)
    self.last = []
    # 512，17，256
    self.U_map = torch.nn.Linear(33, 1, bias=True)
    torch.nn.init.normal_(self.U_map.weight, 0, 0.01)
    torch.nn.init.normal_(self.U_map.bias, 0, 0.01)
    self.mask =MaskModel()
  
  def init_token_order(self, x, module, scale):
    bs, seqlen, vlen = x.shape
    x = x.reshape(bs, seqlen*scale, vlen//scale)
    #x : 512,16,16
    x_list = []
    for i in range(seqlen*scale):
        x_list.append(module.full_forward(x[:, i, :].unsqueeze(1)))
    x = torch.cat(x_list, -1)

    return x

  def forward(self, x, inputs_list=None):
    """Naive full forward pass."""
    #print("rrrrrrrrrr")
    x = torch.sigmoid(self.input_map(x))
    x = x.unfold(dimension=-2, size=16, step=1)
    x = x.permute(0, 1, 3, 2)
    x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
    #print("gggggggg")
    
    # 我们的自适应
    x = self.mask(x)
    
    
    x = x.permute(0, 2, 1)
    #线性层
    x = self.U_map(x)
    x = x.permute(0, 2, 1)
    x = torch.nn.functional.gelu(x)


    #x : 512,1,256
    #print("9999999999999")
    if len(self.last) == 0:
      #print("555555")
      for i, layer in enumerate(self.layers):
          if i % 2 == 0:
            x = self.init_token_order(x, layer, self._scale//(pow(2, i//2)))
            self.last.append(x[:, :, self._vocab_dim*(pow(2, i//2)):].detach())
          else:
            #print(i)
            x = layer.full_forward(x)
    else:
      for i, layer in enumerate(self.layers):
          #print("111111111")
          if i % 2 == 0:
            #print("xxxxxxxxxxxxxxxxxxx")
            new_token = layer.full_forward(x[:, :, -self._vocab_dim*(pow(2, i//2)):])
            x = torch.cat([self.last[i//2], new_token], dim=-1)
            self.last[i//2] = x[:, :, self._vocab_dim*(pow(2, i//2)):].detach() 
          else:
            #print("cccccccc")
            x = layer.full_forward(x)


    #x = x.transpose(1,2)
    # x = self.X_map(x)
    # x = torch.nn.functional.gelu(x)  
    # #x = x * torch.nn.functional.sigmoid(x)
    # x = self.Z_map(x)
    # #x = x.transpose(1,2)
    x = self.output_logit_map(x)
    return x

  def full_loss(self,
                inputs,
                with_grad=True,inputs_lista=None,streamPoolingb=None,
                nonpad_mask=None,
                return_acc=False):
    """Naive full loss and grad."""
    # if inputs_list is not None:
    #   for i in range(1,4):
    #     with torch.cuda.stream(streamPooling[i]):
    #       inputs_list[i] = inputs_list[i].to("cuda",non_blocking=True).long()
    global inputs_list
    inputs_list=inputs_lista
    global streamPooling
    streamPooling=streamPoolingb
    global streamcount 
    streamcount= 0

    logits = self.forward(inputs[:, :-1])
    logits = logits.transpose(1, 2)
    loss = torch.nn.functional.cross_entropy(
            logits[:, :, -1], inputs[:, -1], reduction='mean')
  
    if with_grad:
      loss.backward()

    return loss, logits
    

class dense_baens(nn.Module):
  def __init__(self, N=5, B=4, D1=3, D2=2):
    super(dense_baens, self).__init__()

    self.N = N
    self.B = B
    self.D1 = D1
    self.D2 = D2
    self.U = nn.Parameter(torch.normal(0, 0.01, (N, D1, D2)), requires_grad=True)
    self.bias = nn.Parameter(torch.normal(0, 0.01, (N, B, D2)), requires_grad=True)

  def forward(self, x):
    act = torch.bmm(x, self.U)
    act += self.bias
    return act

class BELayer(torch.nn.Module):

  def __init__(self, branch, vocab_dim, ffn_dim, batch_size, ea=[True, True], trans=False):

    super(BELayer, self).__init__()
    self.branch = branch
    self.vocab_dim = vocab_dim
    self.ffn_dim = ffn_dim
    self.batch_size = batch_size
    self.V_map = dense_baens(batch_size, branch, vocab_dim, vocab_dim)
    self.layernorm1 = torch.nn.LayerNorm(vocab_dim, eps=1e-05, elementwise_affine=ea[0])
    self.layernorm2 = torch.nn.LayerNorm(vocab_dim, eps=1e-05, elementwise_affine=ea[1])
    self.trans = trans
    self.ln1 = 1

  def full_forward(self, x):
    x = x.reshape(self.batch_size, self.branch, self.vocab_dim)

    if self.ln1:
      x = self.layernorm1(x)
    skip = x
    #print("zzzzzzzzzzz")
    # global streamcount
    # global inputs_list
    # global streamPooling
    #print("stream:"+str(streamcount))
    x = self.V_map(x)
    # if streamcount == 0:
    #   #print("inputlista:")
    #   #print(inputs_list)
    #   streamcount=streamcount+1
    #   if inputs_list is not None:
    #     for i in range(1,4):
    #       with torch.cuda.stream(streamPooling[i]):
    #         inputs_list[i] = inputs_list[i].to("cuda",non_blocking=True).long()
    x = self.layernorm2(x)
    x = torch.nn.functional.gelu(x)
    x = (skip + x)/2
    x = x.reshape(self.batch_size, 1, self.branch*self.vocab_dim)

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
    #print("vvvvvv")
    # global streamcount
    # global inputs_list
    # global streamPooling
    # #print("stream:"+str(streamcount))
    # if streamcount == 0:
    #   #print("inputlistb:")
    #   #print(input_list)
    #   streamcount=streamcount+1
    #   if inputs_list is not None:
    #     for i in range(1,4):
    #       with torch.cuda.stream(streamPooling[i]):
    #         inputs_list[i] = inputs_list[i].to("cuda",non_blocking=True).long()
    x = self._ffn(x)

    return x

  def _ffn(self, x):
    
    if self.ln1:
      x = self.layernorm1(x)
         
    skip = x
    
    global streamcount
    global inputs_list
    global streamPooling
    #print("stream:"+str(streamcount))
    if streamcount == 0:
      #print("inputlistb:")
      #print(input_list)
      streamcount=streamcount+1
      if inputs_list is not None:
        for i in range(1,4):
          with torch.cuda.stream(streamPooling[i]):
            inputs_list[i] = inputs_list[i].to("cuda",non_blocking=True).long()
    
    x = self.U_map(x)

    x = torch.nn.functional.gelu(x)
    x = self.V_map(x)
    
    x = self.layernorm2(x)
    x = torch.nn.functional.gelu(x)
    
    x = (skip + x)/2


    return x
