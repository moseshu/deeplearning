#!/bin/bash
data_dir=/DATA/jupyter/personal/translation
data=$data_dir/data/ja-zhbpe
#rm -rf data-bin/
#fairseq-preprocess --source-lang ja --target-lang zh \
#    --trainpref $data/train  \
#    --validpref $data/valid \
#    --testpref $data/test \
#    --destdir data-bin/train.ja-zh \
#    --workers 20


#mkdir -p checkpoints/iwslt_ja_zh
fairseq-train \
    ${data_dir}/data-small/train.ja-zh \
    --arch transformer_iwslt_de_en --max-epoch 100 \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 200 \
    --save-dir /model_dir \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --distributed-world-size 1 \
    --ddp-backend=legacy_ddp \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --encoder-attention-heads 8 \
    --encoder-layers 24 \
    --decoder-attention-heads 8 \
    --decoder-layers 24 

