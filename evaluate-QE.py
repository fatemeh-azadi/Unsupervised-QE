import numpy as np
import subprocess
from argparse import ArgumentParser
import pytorch_lightning as pl
import os 
import transformers
import itertools
import torch
import matplotlib.pyplot as plt
from mosestokenizer import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

eval_script_add = "qe-eval-scripts/sent_evaluate.py"

tokenize = MosesTokenizer('en')
detokenize = MosesDetokenizer('en')

def get_XLMRScore(model, tokenizer, src_sent, tgt_sent, layers = 13):
  sent_src = src_sent.strip().split()
  sent_tgt = tgt_sent.strip().split()
  token_src = [tokenizer.tokenize(word) for word in sent_src]
  token_tgt = [tokenizer.tokenize(word) for word in sent_tgt]
  wid_src = [tokenizer.convert_tokens_to_ids(x) for x in token_src]
  wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
  ids_src = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids']
  ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids']

  model.eval()
  Ps = []
  Rs = []
  Fs = []
  with torch.no_grad():
    out_src_all = model(ids_src.unsqueeze(0), output_hidden_states=True)[2]
    out_tgt_all = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2]

    for align_layer in range(0, layers):
      out_src = out_src_all[align_layer][0, 1:-1]
      out_tgt = out_tgt_all[align_layer][0, 1:-1]
      out_src.div_(torch.norm(out_src, dim=-1).unsqueeze(-1))
      out_tgt.div_(torch.norm(out_tgt, dim=-1).unsqueeze(-1))
      sim = torch.matmul(out_src, out_tgt.transpose(-1, -2))
      
      precision = sim.max(dim=0)[0]
      recall = sim.max(dim=1)[0]

      precision = precision.sum(dim=0) / precision.size(dim=0)
      recall = recall.sum(dim=0) / recall.size(dim=0)
      
      F = 2 * precision * recall / (precision + recall)
      Ps.append(precision.item())
      Rs.append(recall.item())
      Fs.append(F.item())
  return (Ps, Rs, Fs)

def get_scores(data_src, data_tgt, model, tokenizer, layers = 13):
 
    outP = []
    outR = []
    outF = []
    for line_src, line_tgt in zip(data_src, data_tgt):
        (P, R, F) = get_XLMRScore(model, tokenizer, line_src, line_tgt, layers)
        outP.append(P)
        outR.append(R)
        outF.append(F)
    return outP, outR, outF

def get_scores_with_unk(data_src, data_tgt, model, tokenizer, vocab, layers = 13):
 
    outP = []
    outR = []
    outF = []
    for line_src, line_tgt in zip(data_src, data_tgt):
        line_tgt = replace_unk(line_tgt, vocab)
        (P, R, F) = get_XLMRScore(model, tokenizer, line_src, line_tgt, layers)
        outP.append(P)
        outR.append(R)
        outF.append(F)
    return outP, outR, outF


def read_vocab(vocab_path):
   vocab = set()
   with open(vocab_path, "r", encoding="utf-8") as vocabFile:
     for v in vocabFile:
       vocab.add(v.strip())
   return vocab


def replace_unk(line_tgt, vocab):
   line = tokenize(line_tgt.strip())
   line2 = []
   for w in line:
      if(w in vocab or w.lower() in vocab):
         line2.append(w)
      else:
         line2.append("<unk>")
   line2 = detokenize(line2)
   return line2

def save_scores(scores, out_path, out_str, layers = 13):
   
    for i in range(0, layers):
        out_file = open(f"{out_path}-{i}", "w")
        for j in range(0, len(scores)):
            out_file.write(f"{out_str}\t{j}\t{scores[j][i]}\n")
        out_file.close()

def read_data(in_add):
    in_file = open(in_add, "r", encoding="utf-8")
    data = []
    for line in in_file:
        data.append(line)
    return data

if __name__=="__main__":

    parser = ArgumentParser()

    parser.add_argument("--model", default='xlm-roberta-base', type=str, help="the base model (e.g. xlm-roberta-base)")

    parser.add_argument("--src_add", type=str, required=True, help="source file address")
    parser.add_argument("--tgt_add", type=str, required=True, help="target file address")
    parser.add_argument("--gold_add", type=str, required=True, help="gold alignments address")
    parser.add_argument("--vocab", default='', type=str, help="target vocabulary file address")
    parser.add_argument("--langs", default="en-fa", type=str, help="language pair (e.g. en-fa)")
    parser.add_argument("--align_layer", default=9, type=int, help="the layer the word representations should be taken from")
    parser.add_argument("--out_dir", type=str, required=True, help="output folder containing QE scores files (i.e., Precision, Recall and F1 scores)")
    parser.add_argument("--sys_name", default='xlmr-base', type=str, help="output file names prefix")

    hparams = parser.parse_args()

    model = transformers.AutoModel.from_pretrained(hparams.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")

    data_src = read_data(hparams.src_add)
    data_tgt = read_data(hparams.tgt_add)
    
    if(hparams.vocab == ''):
        (P, R, F) = get_scores(data_src, data_tgt, model, tokenizer)
    else:
        vocab = read_vocab(hparams.vocab)
        (P, R, F) = get_scores_with_unk(data_src, data_tgt, model, tokenizer, vocab)

    save_scores(P, f"{hparams.out_dir}/{hparams.langs}/{hparams.sys_name}-Precision", 
                f"{hparams.langs}\t{hparams.sys_name}-Precision")
    save_scores(R, f"{hparams.out_dir}/{hparams.langs}/{hparams.sys_name}-Recall", 
                f"{hparams.langs}\t{hparams.sys_name}-Recall")
    save_scores(F, f"{hparams.out_dir}/{hparams.langs}/{hparams.sys_name}-F1", 
                f"{hparams.langs}\t{hparams.sys_name}-F1")


    for layer in range(0, 13):
        out_add = f"{hparams.out_dir}/{hparams.langs}/{hparams.sys_name}-Precision-{layer}"

        precision = subprocess.check_output(f"python {eval_script_add} {out_add} {hparams.gold_add} | grep \"pearson:\" | sed \"s/pearson: //g\"", 
                                    shell=True)
        out_add = f"{hparams.out_dir}/{hparams.langs}/{hparams.sys_name}-Recall-{layer}"
        recall = subprocess.check_output(f"python {eval_script_add} {out_add} {hparams.gold_add} | grep \"pearson:\" | sed \"s/pearson: //g\"", 
                                    shell=True)
        out_add = f"{hparams.out_dir}/{hparams.langs}/{hparams.sys_name}-F1-{layer}"
        f1 = subprocess.check_output(f"python {eval_script_add} {out_add} {hparams.gold_add} | grep \"pearson:\" | sed \"s/pearson: //g\"", 
                                    shell=True)
        precision = precision.strip().decode('utf-8')
        recall = recall.strip().decode('utf-8')
        f1 = f1.strip().decode('utf-8')
        print(f"Layer {layer}: Precision-Pearson={precision} , Recall-Pearson={recall} , F1-Pearson={f1}")

