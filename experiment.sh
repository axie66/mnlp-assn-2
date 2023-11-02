#!/bin/bash


# Preprocess Guarani data
PYTHONPATH=./ python3 examples/speech_to_text/prep_librispeech_data.py \
    --output-root cv-corpus/gn/ \
    --vocab-type unigram \
    --vocab-size 2000


# Train LSTM
# 54,167,552 params
fairseq-train cv-corpus/gn \
    --save-dir lstm_output \
    --config-yaml config.yaml \
    --train-subset train_manifest  \
    --valid-subset dev_manifest \
    --num-workers 4 \
    --max-tokens 10000 \
    --max-epoch 100 \
    --task speech_to_text \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --report-accuracy \
    --arch s2t_berard_512_5_3 \
    --optimizer adam \
    --lr 2e-3 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 100 \
    --clip-norm 10.0 \
    --seed 1 \
    --update-freq 32 \
    --save-interval 10
# Generate with LSTM
fairseq-generate cv-corpus/gn \
    --config-yaml config.yaml \
    --gen-subset dev_manifest_gn \
    --task speech_to_text --path lstm_output/checkpoint_last.pt \
    --max-tokens 50000 --scoring wer


# Train Conformer
# 52,933,376 params
fairseq-train cv-corpus/gn \
    --save-dir s2t_conformer_output \
    --config-yaml config.yaml \
    --train-subset train_manifest  \
    --valid-subset dev_manifest \
    --num-workers 4 \
    --max-tokens 10000 \
    --max-epoch 100 \
    --task speech_to_text \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --report-accuracy \
    --arch s2t_conformer \
    --pos-enc-type rope \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --lr 2e-3 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 100 \
    --clip-norm 10.0 \
    --seed 1 \
    --update-freq 32 \
    --save-interval 10
# Generate with Conformer
fairseq-generate cv-corpus/gn \
    --config-yaml config.yaml \
    --gen-subset dev_manifest_gn \
    --task speech_to_text --path s2t_conformer_output/checkpoint_last.pt \
    --max-tokens 50000 --scoring wer


# Train Speech2Text
# 27,488,256 params
fairseq-train cv-corpus/gn \
    --save-dir s2t_transformer_s_output \
    --config-yaml config.yaml \
    --train-subset train_manifest  \
    --valid-subset dev_manifest \
    --num-workers 4 \
    --max-tokens 40000 \
    --max-epoch 100 \
    --task speech_to_text \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --report-accuracy \
    --arch s2t_transformer_s \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --lr 2e-3 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 100 \
    --clip-norm 10.0 \
    --seed 1 \
    --update-freq 8 \
    --save-interval 10
# Generate with Speech2Text
fairseq-generate cv-corpus/gn \
    --config-yaml config.yaml \
    --gen-subset dev_manifest_gn \
    --task speech_to_text --path s2t_transformer_s_output/checkpoint_last.pt \
    --max-tokens 50000 --scoring wer


# Preprocess multlingual data
# ES and PT data must first be downloaded to cv-corpus/
# Spanish
PYTHONPATH=./ python3 examples/speech_to_text/prep_librispeech_data.py \
    --output-root cv-corpus/es/,cv-corpus/gn/,data_out_es \
    --vocab-type unigram \
    --vocab-size 2000
# Portuguese
PYTHONPATH=./ python3 examples/speech_to_text/prep_librispeech_data.py \
    --output-root cv-corpus/pt/,cv-corpus/gn/,data_out_pt \
    --vocab-type unigram \
    --vocab-size 2000


# Conformer multilingual training
# Spanish
fairseq-train data_out_es \
    --save-dir es_s2t_conformer_output \
    --config-yaml config.yaml \
    --train-subset train_manifest_es,train_manifest_gn  \
    --valid-subset dev_manifest_gn \
    --num-workers 4 \
    --max-tokens 10000 \
    --max-epoch 100 \
    --task speech_to_text \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --report-accuracy \
    --arch s2t_conformer \
    --pos-enc-type rope \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --lr 2e-3 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 100 \
    --clip-norm 10.0 \
    --seed 1 \
    --update-freq 32 \
    --save-interval 10
# Generate with Conformer
fairseq-generate data_out_es \
    --config-yaml config.yaml \
    --gen-subset dev_manifest_gn \
    --task speech_to_text --path es_s2t_conformer_output/checkpoint_last.pt \
    --max-tokens 50000 --scoring wer

# Portuguese
fairseq-train data_out_pt \
    --save-dir pt_s2t_conformer_output \
    --config-yaml config.yaml \
    --train-subset train_manifest_pt,train_manifest_gn  \
    --valid-subset dev_manifest_gn \
    --num-workers 4 \
    --max-tokens 10000 \
    --max-epoch 100 \
    --task speech_to_text \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --report-accuracy \
    --arch s2t_conformer \
    --pos-enc-type rope \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --lr 2e-3 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 100 \
    --clip-norm 10.0 \
    --seed 1 \
    --update-freq 32 \
    --save-interval 10
# Generate with Conformer
fairseq-generate data_out_pt \
    --config-yaml config.yaml \
    --gen-subset dev_manifest_gn \
    --task speech_to_text --path pt_s2t_conformer_output/checkpoint_last.pt \
    --max-tokens 50000 --scoring wer
