export CODE_DIR=crosslingual-nlp

# The parallel data with their giza++ alignments should be stored in `ALIGNED_DATA_DIR` as:
# - /$lang-pair/Parallel.$src
# - /$lang-pair/Parallel.$tgt
# - /$lang-pair/aligned.intersection

export ALIGNED_DATA_DIR=`pwd`/data/aligned-data

# Set envrionment variable `ROOT_DIR`, all related files will be store in this directory.
export ROOT_DIR=`pwd`/Fine-Tune/
mkdir $ROOT_DIR
export DATA_DIR=$ROOT_DIR/dataset
export CKPT_DIR=$ROOT_DIR/checkpoints/crosslingual-nlp

cd $CODE_DIR

chmod -R +x tools
outdir="$DATA_DIR"/bitext/clean/
mkdir -p "$outdir"

lg_pairs="en-fa en-hi ps-en km-en si-en"

for pair in $lg_pairs; do
  lgs=($(echo $pair | tr "-" "\n"))
  src=${lgs[0]}
  tgt=${lgs[1]}

  out="$outdir"/"$pair"

  cp "$ALIGNED_DATA_DIR"/"$pair"/Parallel."$src" "$out".$src
  cp "$ALIGNED_DATA_DIR"/"$pair"/Parallel."$tgt" "$out"."$tgt"
  cp "$ALIGNED_DATA_DIR"/"$pair"/aligned.intersection "$out".align

  nb_val=10000

  if [ -f "$out".val.align ]; then
    echo file exists
    exit
  fi

  # concat
  if [ ! -f "$out".pair ]; then
    python scripts/bitext-concat.py "$out".$src "$out"."$tgt" >"$out".pair
  fi


  if [ ! -f "$out".val.align ]; then
    python scripts/bitext-split.py "$out".pair "$out".align "$nb_val" "$out"
  fi
done
