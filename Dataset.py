import torch
import nltk
import pickle
import re

from nltk import word_tokenize
from torch.utils.data import Dataset
from datasets import load_dataset
from gensim.models import Word2Vec
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






class OpusBook(Dataset):
  def __init__(self,size,device,save= False):
    self.size    = size
    self.save    = save
    self.device  = device
    self.data    = self.data_loading(500)
    self.src,self.trg,self.src_tokens,self.trg_tokens = self.preprocessing(self.data,self.save)
    self.src_vectors,self.trg_vectors                 = self.modelization([self.src_tokens,self.trg_tokens])
    self.src_vocab = len(self.src_vectors)
    self.trg_vocab = len(self.trg_vectors)
    self.src_sos   = self.src_vocab + 1
    self.src_eos   = self.src_vocab + 2
    self.src_uis   = self.src_vocab + 3
    self.trg_sos   = self.trg_vocab + 1
    self.trg_eos   = self.trg_vocab + 2
    self.trg_uis   = self.trg_vocab + 3
    self.seq_len   = 350

  def data_loading(self,size):
    books = load_dataset("opus_books", "en-fr")
    nltk.download('punkt')
    data = []
    for i in range(self.size):
      data.append(books["train"][i])
    return data
  def preprocessing(self,data,save):
    english = []
    french  = []
    eng_tokens = []
    frn_tokens = []
    if save:
      with open("Weights/english_preprocess","rb") as f:
        english = pickle.load(f)
      with open("Weights/french_preprocess","rb") as f:
        french =  pickle.load(f)

      with open("Weights/eng_tokens","rb") as f:
        eng_tokens = pickle.load(f)

      with open("Weights/frn_tokens","rb") as f:
        frn_tokens = pickle.load(f)

    if not save:
      for example in data:
        eng = re.sub(r'[^\w\s]',"",example["translation"]["en"])
        english.append(eng)
        frn = re.sub(r'[^\w\s]',"",example["translation"]["fr"])
        french.append(frn)
      #pickeling
      with open("Weights/english_preprocess","wb") as f:
        pickle.dump(english,f)
      with open("Weights/french_preprocess","wb") as f:
        pickle.dump(french,f)
      #tokenization


      for i in range(len(english)):
        tokens = word_tokenize(english[i])
        eng_tokens.append(tokens)
        tokens = word_tokenize(french[i])
        frn_tokens.append(tokens)

      with open("Weights/eng_tokens","wb") as f:
        pickle.dump(eng_tokens,f)

      with open("Weights/frn_tokens","wb") as f:
        pickle.dump(frn_tokens,f)

    return english,french,eng_tokens,frn_tokens

  def modelization(self,tokens):
    eng_model = Word2Vec(tokens[0],min_count = 1)
    frn_model = Word2Vec(tokens[1],min_count = 1)
    return eng_model.wv.key_to_index,frn_model.wv.key_to_index

  def vectroization(self,indexes: list,data:list):
    eng = []
    frn = []
    english = word_tokenize(data[0])
    french = word_tokenize(data[1])
    for i in range(len(english)):
      eng.append(indexes[0][english[i]])
    for i in range(len(french)):
      frn.append(indexes[1][french[i]])
    eng = torch.tensor(eng).int().to(self.device)
    frn = torch.tensor(frn).int().to(self.device)
    return eng,frn

  def sentence_combine(self,src,trg):
    input_len = self.seq_len - len(src)-2
    output_len = self.seq_len - len(trg)-2
    if len(src) >= self.seq_len or len(trg) >= self.seq_len:
      x = max(len(src),len(trg))
      i = [len(src),len(trg)].index(x)
      if i ==0:
        input_len = 0
        output_len = x- len(trg)
      else:
        output_len = 0
        input_len  = x- len(src)


    src = torch.cat([
        torch.tensor([self.src_sos],dtype = torch.int32),
        src,
        torch.tensor([self.src_eos],dtype = torch.int32),
        torch.tensor([self.src_uis]*input_len,dtype = torch.int32),
    ])
    trg = torch.cat([
        torch.tensor([self.trg_sos],dtype = torch.int32),
        trg,
        torch.tensor([self.trg_eos],dtype = torch.int32),
        torch.tensor([self.trg_uis]*output_len,dtype = torch.int32),
    ])
    return src,trg

  def __len__(self):
    return len(self.src)

  def __getitem__(self,idx):
    eng = self.src[idx]
    frn = self.trg[idx]

    src,trg = self.vectroization([self.src_vectors,self.trg_vectors],[eng,frn])

    src_v,trg_v = self.sentence_combine(src,trg)

    return eng,frn,src_v,trg_v