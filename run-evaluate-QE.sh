

langs="en-fa"

mkdir -p outputs-QE-new/$langs

python evaluate-QE.py \
 --src_add data/V3/input.txt \
 --tgt_add  data/V3/mt-output.txt \
 --gold_add data/V3/TER.txt2 \
 --out_dir outputs-QE-new/ \
 --langs $langs --sys_name XLMR-$langs \
 --model xlm-roberta-base \
 > outputs-QE-new/XLMR-$langs
