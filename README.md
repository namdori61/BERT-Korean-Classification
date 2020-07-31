# BERT-Korean-Classification
Various Korean BERTs classification experiment.

## Goal
NSMC 데이터셋을 이용하여, 한국어 관련 3가지 BERT의 Binary Classification (Sentiment Analysis) 성능 비교해보기

## Dataset
Naver sentiment movie corpus v1.0 : https://github.com/e9t/nsmc

## Pre-trained BERT
1. BERT-multilingual : https://huggingface.co/transformers/model_doc/bert.html#overview
2. SKT KoBERT : https://github.com/SKTBrain/KoBERT
3. Beomi(이준범 님) KcBERT : https://github.com/Beomi/KcBERT

## How to execute codes
 - `Preprocess: python preprocess.py --input_path INPUT_PATH(txt file) --output_path OUTPUT_PATH(jsonl file)`
 - `Train: python train.py --input_path INPUT_PATH(jsonl file) --model BERT --cuda_device GPU_NUM
 --max_epochs MAX_EPOCHS --save_dir SAVE_DIR --batch_size BATCH_SIZE --lr LEARNING_RATE --num_workers NUM_WORKERS`