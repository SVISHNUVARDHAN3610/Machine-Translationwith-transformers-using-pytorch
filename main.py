import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("./")

from Dataset import OpusBook
from Network import Embedding,MultiHeadAttention,Encoder,Decoder,EncoderBlock,DecoderBlock,Transformer,FeedForward,ProjectionLayer,device



data = OpusBook(127085,"cpu",False)

src_vocab  = data.src_vocab+4
trg_vocab  = data.trg_vocab+4
lr         = 0.000084
d_model    = 512
heads      = 8
src_embedding = Embedding(src_vocab,d_model)
trg_embedding = Embedding(trg_vocab,d_model)



encoder_blocks = []
for i in range(heads):
  enc_attention     = MultiHeadAttention(d_model,heads)
  enc_network       = FeedForward(d_model)
  encoder_block     = EncoderBlock(d_model,enc_attention,enc_network)
  encoder_blocks.append(encoder_block)

decoder_blocks = []
for i in range(heads):
  dec_attention     = MultiHeadAttention(d_model,heads)
  dec_network       = FeedForward(d_model)
  decoder_block     = DecoderBlock(d_model,enc_attention,enc_network)
  decoder_blocks.append(decoder_block)

encoder = Encoder(encoder_blocks)
decoder = Decoder(decoder_blocks)
projection = ProjectionLayer(d_model,trg_vocab)

transformer = Transformer(d_model,encoder,decoder,projection,src_embedding,trg_embedding)

optimizer   = optim.Adam(transformer.parameters() ,lr = lr)
loss_en    = nn.CrossEntropyLoss()



transformer.load_state_dict(torch.load("Weights/transformer.pth"))


trg_rev_ve ={value: key for key, value in data.trg_vectors.items()}

trg_rev_ve.update({data.trg_vocab+1:"<sos>"})
trg_rev_ve.update({data.trg_vocab+2:"<eos>"})
trg_rev_ve.update({data.trg_vocab+3:"<uis>"})


def reverse(tokens):
  main = ""
  for x in tokens:
    word = trg_rev_ve[x.item()]
    main += word
    main += " "
  return main




epoch = len(data)
loss_da = []
last = 97170
epi = []
print("cuda",torch.cuda.is_available())
for i in range(last,epoch):
  epi.append(i)
  src,trg,src_v,trg_v = data[i]
  src_v,trg_v = src_v.to(device),trg_v.to(device)
  pred = transformer(src_v,trg_v).argmax(1)

  loss = loss_en(trg_v.float(),pred.float())
  #print("===============================================================","\n")
  # ram = psutil.virtual_memory()[3]/1000000000
  # cpu = psutil.cpu_percent(4)
  print("episode: {}/{} , percentage : {}% , loss: {} , cuda : {} ".format(i,epoch,int((i/epoch)*100),loss.item(),torch.cuda.is_available()),"\n")
#   print("english: {}".format(src),"\n")
#   print("french actual: {}".format(trg),"\n")
#   print("french predected: {}".format(reverse(pred)),"\n")
  loss_da.append(loss.item())
  loss.requires_grad = True
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if i %30 ==0:
    transformer.load_state_dict(torch.load("Weights/transformer.pth"))
    torch.save(transformer.state_dict(),"Weights/transformer.pth")
  plt.plot(epi,loss_da)
  plt.savefig("Weights/transformer.png")
  plt.close()
with open("loss","wb") as f:
  pickle.dump(loss_da,f)
