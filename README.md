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
 - `Train: python train.py --input_path INPUT_PATH(jsonl file) --model BERT [BERT, KoBERT, KcBERT]
 --version lr_LR_wm_WARMUP_WEIGHTDECAY --lr LR --warm_up WARMUP --weight_decay WEIGHTDECAY --cuda_device GPU_NUM
 --max_epochs MAX_EPOCHS --save_dir SAVE_DIR --batch_size BATCH_SIZE --lr LEARNING_RATE --num_workers NUM_WORKERS`
 
 ## Performances (Validation)
 
| Model  | Epochs | Accuracy | Loss | 
| :-------------: | :-------------: | :-------------: | :-------------: |
| BERT-multilingual  | 2  | 0.8488  | 0.3354  |
| SKT KoBERT  |   |   |  |
| Beomi's KcBERT  | 2  | 0.8952  | 0.2548  |
 
 ## Lessons learned
 1. Transformer 기반 크기가 큰 모델에서 warm-up learning rate scheduling 은 학습에 매우! 중요하다! 이거로 loss가 수렴하냐 안하냐를 결정지음.
 2. Transfer learning 의 fine tuning 단계에서는 매우 작은 learning rate를 사용해야 한다! 