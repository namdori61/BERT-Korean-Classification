import json
import logging
from typing import Any, Dict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import gluonnlp as nlp

logger = logging.getLogger(__name__)


class KoBertDataset(Dataset):
    def __init__(self,
                 file_path: str = None,
                 tokenizer: Any = None,
                 max_length: int = 140):

        logger.info(f'Reading file at {file_path}')

        with open(file_path) as dataset_file:
            self.dataset = dataset_file.readlines()

        logger.info('Reading the dataset')


        self.processed_dataset = []

        transform = nlp.data.BERTSentenceTransform(tokenizer=tokenizer,
                                                   max_seq_length=max_length,
                                                   pad=True,
                                                   pair=False)

        def gen_attention_mask(token_ids, valid_length):
            attention_mask = torch.zeros_like(token_ids)
            for i in range(valid_length.item()):
                attention_mask[i] = 1
            return attention_mask

        for line in tqdm(self.dataset, desc='Processing'):
            data = json.loads(line)
            processed_data = {}
            try:
                encoded_data = transform([data['document']])
            except:
                encoded_data = transform([' '])
            processed_data['input_ids'] = torch.LongTensor(encoded_data[0])
            processed_data['attention_mask'] = gen_attention_mask(processed_data['input_ids'], encoded_data[1])

            processed_data['label'] = torch.LongTensor([int(data['label'])])

            self.processed_dataset.append(processed_data)

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self,
                    idx: int = None):
        return self.processed_dataset[idx]