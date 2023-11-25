import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Embedding
class Embedding(nn.Module):
  def __init__(self,vocab_size,d_model):
    super().__init__()
    self.vocab_size = vocab_size
    self.d_model    = d_model
    self.embedding  = nn.Embedding(self.vocab_size,self.d_model).to(device)

  def forward(self,x):
    x = self.embedding(x)
    return x/math.sqrt(self.d_model)
  

  #MultiHead Attention
class MultiHeadAttention(nn.Module):
  def __init__(self,d_model,heads):
    super().__init__()
    self.d_model = d_model
    self.heads   = heads
    self.d_k     = self.d_model//self.heads

    self.dropout = nn.Dropout(0.5)
    self.w_q     = nn.Linear(self.d_model,self.d_model).to(device)
    self.w_k     = nn.Linear(self.d_model,self.d_model).to(device)
    self.w_v     = nn.Linear(self.d_model,self.d_model).to(device)
    self.w_o     = nn.Linear(self.d_model,self.d_model).to(device)
  def attention(self,q,k,v,drop = True):
    d_k = q.shape[-1]
    attention_scores = q @k.transpose(-2,-1)/math.sqrt(d_k)
    attention_scores = f.softmax(attention_scores)
    if drop:
      attention_scores = self.dropout(attention_scores)
    attention = attention_scores @v
    return attention, attention_scores
  def forward(self,q,k,v):
    query = self.w_q(q)
    key = self.w_k(k)
    value = self.w_v(v)
    attention,score = self.attention(query,key,value)
    attention = attention.contiguous()
    return self.w_o(attention)
  


class ResidualConnection(nn.Module):
  def __init__(self,d_model):
    super().__init__()
    self.d_model = d_model
    self.norm    = nn.LayerNorm(self.d_model).to(device)
    self.dropout = nn.Dropout(0.5)

  def forward(self,x,sublayer):
    x = x + self.dropout(sublayer(x))
    return self.norm(x)
  
class FeedForward(nn.Module):
  def __init__(self,d_model):
    super().__init__()
    self.d_model = d_model
    self.layer1 = nn.Linear(self.d_model,self.d_model*4).to(device)
    self.layer2 = nn.Linear(self.d_model*4,self.d_model).to(device)
    self.dropout= nn.Dropout(0.5)

  def forward(self,x):
    x = f.relu(self.layer1(x))
    x = self.dropout(self.layer2(x))
    return x

class EncoderBlock(nn.Module):
  def __init__(self,d_model,attention,feedforward):
    super().__init__()
    self.attention    = attention
    self.network      = feedforward
    self.residual     = nn.ModuleList([ResidualConnection(d_model) for _ in range(2)])

  def forward(self,x):
    x = self.residual[0](x, lambda x: self.attention(x,x,x))
    x = self.residual[1](x, lambda x: self.network(x))
    return x



class Encoder(nn.Module):
  def __init__(self,layers):
    super().__init__()
    self.layers = layers
  def forward(self,x):
    for layer in self.layers:
      x = layer(x)
    return x

class Decoder(nn.Module):
  def __init__(self,layers):
    super().__init__()
    self.layers = layers
  def forward(self,encoder,x):
    for layer in self.layers:
      x = layer(encoder,x)
    return x

class DecoderBlock(nn.Module):
  def __init__(self,d_model,attention,feedforward):
    super().__init__()
    self.attention = attention
    self.network   = feedforward
    self.residual     = nn.ModuleList([ResidualConnection(d_model) for _ in range(3)])

  def forward(self,encoder_output,x):
    x = self.residual[0](x , lambda x: self.attention(x,x,x))
    x = self.residual[1](x , lambda x: self.attention(encoder_output,encoder_output,x))
    x = self.residual[2](x , lambda x: self.network(x))
    return x


class ProjectionLayer(nn.Module):
  def __init__(self,d_model,trg_vocab):
    super().__init__()
    self.linear = nn.Linear(d_model,trg_vocab).to(device)

  def forward(self,x):
    x = f.softmax(self.linear(x))
    return x

class Transformer(nn.Module):
  def __init__(self,d_model,encoder,decoder,projection,src_embed,trg_embed):
    super().__init__()
    self.d_model = d_model
    self.encoder = encoder
    self.decoder = decoder
    self.projection    = projection
    self.src_embedding = src_embed
    self.trg_embedding = trg_embed
    self.init_weights()
  def init_weights(self):
    for m in self.modules():
      if isinstance(m,nn.Embedding):
        nn.init.normal_(m.weight,mean = 0.0,std = 1.0)
      elif isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
      elif isinstance(m,nn.LayerNorm):
        nn.init.zeros_(m.weight)
  def positional_encoding(self,x):
    pe = torch.zeros(len(x), self.d_model)
    position = torch.arange(0, len(x), dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe

  def forward(self,src,trg):
    src = self.src_embedding(src)
    trg = self.trg_embedding(trg)
    src_pe = self.positional_encoding(src)
    trg_pe = self.positional_encoding(trg)
    src = src#+ src_pe
    trg = trg#+ trg_pe
    encode = self.encoder(src)
    decode = self.decoder(encode,trg)
    pred   = self.projection(decode)
    return pred

