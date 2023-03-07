#!/bin/bash
data_dir=/DATA/jupyter/personal/translation
data=$data_dir/data/ja-zhbpe
#rm -rf data-bin/
fairseq-preprocess --source-lang ja --target-lang zh \
    --trainpref $data/train  \
    --validpref $data/valid \
    --testpref $data/test \
    --destdir data-bin/train10.ja-zh \
    --workers 20

