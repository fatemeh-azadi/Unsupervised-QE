# Assuming this repo is save at `/path/to/crosslingual-nlp`
export CODE_DIR=crosslingual-nlp/
export CURL_CA_BUNDLE=""
export PYTHONWARNINGS="ignore:Unverified HTTPS request"

# Set envrionment variable `ROOT_DIR`, all related files will be store in this directory.
export ROOT_DIR=`pwd`/Fine-Tune/
mkdir $ROOT_DIR
export DATA_DIR=$ROOT_DIR/dataset
export CKPT_DIR=$ROOT_DIR/checkpoints/crosslingual-nlp

cd $CODE_DIR

lngs=(en-fa en-hi ps-en km-en si-en)

bs=1
lr=1e-4

sim=cntrstv
proj=nn
temp=0.1
norm=True
l2_param_coeff=1.0
l2_src_coeff=0.0
subset=1


model=xlm-roberta-base


root_dir="$ROOT_DIR"
data_path="$root_dir"/dataset/bitext/clean/
save_path="$root_dir"/checkpoints/alignment
cache_path="$root_dir"/cache/clnlp

CUDA_LAUNCH_BLOCKING=1 python src/train.py \
    --task alignment \
    --data_dir "$data_path" \
    --trn_langs "${lngs[@]}" \
    --val_langs "${lngs[@]}" \
    --cache_dataset True \
    --cache_path "$cache_path" \
    --max_trn_len 96 \
    --max_tst_len 96 \
    --pretrain "$model" \
    --batch_size $bs \
    --eval_batch_size $bs \
    --learning_rate $lr \
    --adam_beta2 0.999 \
    --schedule linear \
    --max_steps 100000 --warmup_steps 4000 --val_check_interval 5000 \
    --input_dropout 0.0 \
    --aligner_sim "$sim" \
    --aligner_projector $proj \
    --aligner_temperature $temp \
    --aligner_normalize $norm \
    --aligner_l2_param_coeff $l2_param_coeff \
    --aligner_l2_src_coeff $l2_src_coeff \
    --mix_sampling True \
    --patience 999999 \
    --gpus 1 \
    --precision 16 \
    --accumulate_grad_batches 32 \
    --subset_ratio $subset \
    --default_save_path "$save_path"/"$(echo "${lngs[@]}" | tr ' ' ',')"/ \
    --exp_name exp1 2>&1 | tee Log

