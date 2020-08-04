# BERT-Korean-Classification to NSMC dataset
Various Korean BERTs classification experiment to NSMC (Naver sentiment movie corpus v1.0) dataset

## Goal
NSMC 데이터셋을 이용하여, 한국어 관련 3가지 BERT의 Binary Classification (Sentiment Analysis) 성능 비교해보기

## Dataset
Naver sentiment movie corpus v1.0 : https://github.com/e9t/nsmc

## Pre-trained BERT
1. BERT-multilingual : https://huggingface.co/transformers/model_doc/bert.html#overview
2. SKT KoBERT : https://github.com/SKTBrain/KoBERT
3. Beomi(이준범 님) KcBERT : https://github.com/Beomi/KcBERT

| Model | Train dataset | Max length | Vocab size | 
| :-------------: | :-------------: | :-------------: | :-------------: |
| BERT-multilingual | 104개 언어 위키피디아 코퍼스 | 512 | 119,547 |
| SKT KoBERT | 한국어 위키, 한국어 뉴스 | 512 | 8,002 |
| Beomi's KcBERT | 네이버 뉴스 댓글과 대댓글 | 300 | 30,000 |

## How to execute codes
 - `Preprocess: python preprocess.py --input_path INPUT_PATH(txt file) --output_path OUTPUT_PATH(jsonl file)`
 - `Train: python train.py --input_path INPUT_PATH(jsonl file) --model BERT [BERT, KoBERT, KcBERT]
 --version lr_LR_wm_WARMUP_WEIGHTDECAY --lr LR --warm_up WARMUP --weight_decay WEIGHTDECAY --cuda_device GPU_NUM
 --max_epochs MAX_EPOCHS --save_dir SAVE_DIR --batch_size BATCH_SIZE --lr LEARNING_RATE --num_workers NUM_WORKERS`
 
 ## Performances (Validation)
 
| Model | Epochs | Accuracy | Loss | 
| :-------------: | :-------------: | :-------------: | :-------------: |
| BERT-multilingual | 2 | 0.851 | 0.3354 |
| SKT KoBERT | 2 | 0.8919 | 0.2624 |
| Beomi's KcBERT | 2 | 0.8961 | 0.2548 |

![validation accuracy](https://user-images.githubusercontent.com/20228736/89353708-f428d300-d6f1-11ea-8e97-e60a45f52ccd.png)

![validation loss](https://user-images.githubusercontent.com/20228736/89353694-e96e3e00-d6f1-11ea-88e7-fd30a5400251.png)
 
 ## Lessons learned
 1. Transformer 기반 크기가 큰 모델에서 warm-up learning rate scheduling 은 학습에 매우! 중요하다! 이거로 loss가 수렴하냐 안하냐를 결정지음.
 2. Transfer learning 의 fine tuning 단계에서는 매우 작은 learning rate를 사용해야 한다! 