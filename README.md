# Unsupervised-QE


This repo contains the implementations for the paper [*Mismatching-Aware Unsupervised Translation Quality Estimation For Low-Resource Languages*](https://arxiv.org/abs/2208.00463).


## Installation

- Environment (conda): `environment.yml`


## Cross-lingual alignment of the Pre-trained Model
The implementations for this part are mainly taken from [*this repo*](https://github.com/shijie-wu/crosslingual-nlp), while our modifications are made in it.

### 1. Preparing the data:

The parallel datasets with their giza++ alignments should be first prepared. Then, [*prepareData.sh*](prepareData.sh) should be executed to convert the data to bitext format and split the validation set from it. For more information see [*prepareData.sh*](prepareData.sh).

### 2. Fine-tuning the model:

The [*runAlignments-all.sh*](runAlignments-all.sh) script should be executed to fine-tune the base model, and the checkpoint will be saved in:

> Fine-Tune/checkpoints/alignment/en-fa,en-hi,ps-en,ne-en,km-en,si-en/exp1/version_0/ckpts/*.ckpt

To see the hyper-parameters or intended languages, check [*runAlignments-all.sh*](runAlignments-all.sh).

Finally, convert the checkpoint to a folder accepted by `transformers.AutoModel.from_pretrained` as follows:
```bash
export SCRIPT_DIR=crosslingual-nlp/example/contrastive-alignment
export CKPT_DIR=Fine-Tune/checkpoints/alignment/en-fa,en-hi,ps-en,ne-en,km-en,si-en/exp1/version_0/ckpts
ckpt=$CKPT_DIR/ckpts_epoch=*.ckpt

python $SCRIPT_DIR/dump.py single $ckpt $CKPT_DIR/aligned_encoder
```
The final model will be in `$CKPT_DIR/aligned_encoder`.

## Running the Unsupervised QE method

To run and evaluate the Unsupervised QE method use [*evaluate-QE.py*](evaluate-QE.py) as:

```
usage: 

evaluate-QE.py [-h] [--model MODEL] --src_add SRC_ADD --tgt_add TGT_ADD --gold_add GOLD_ADD [--vocab VOCAB] [--langs LANGS] [--align_layer ALIGN_LAYER] --out_dir OUT_DIR [--sys_name SYS_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         the base model (e.g. xlm-roberta-base)
  --src_add SRC_ADD     source file address
  --tgt_add TGT_ADD     target file address
  --gold_add GOLD_ADD   gold alignments address
  --vocab VOCAB         target vocabulary file address
  --langs LANGS         language pair (e.g. en-fa)
  --align_layer ALIGN_LAYER
                        the layer the word representations should be taken from
  --out_dir OUT_DIR     output folder containing QE scores files (i.e., Precision, Recall and F1 scores)
  --sys_name SYS_NAME   output file names prefix
```


A sample script can be seen in [*run-evaluate-QE.sh*](run-evaluate-QE.sh).