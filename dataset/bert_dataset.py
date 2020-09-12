import json
import logging
from typing import Any
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BertDataset(Dataset):
    def __init__(self,
                 file_path: str = None,
                 tokenizer: Any = None,
                 max_length: int = 140):

        logger.info(f'Reading file at {file_path}')

        with open(file_path) as dataset_file:
            self.dataset = dataset_file.readlines()

        logger.info('Reading the dataset')

        self.processed_dataset = []

        for line in tqdm(self.dataset, desc='Processing'):
            data = json.loads(line)
            processed_data = {}
            encoded_dict = tokenizer.encode_plus(data['document'],
                                                 add_special_tokens=True,
                                                 max_length=max_length,
                                                 truncation=True,
                                                 pad_to_max_length=True,
                                                 return_attention_mask=True,
                                                 return_tensors='pt'
                                                 )
            processed_data['input_ids'] = encoded_dict['input_ids'].squeeze(0)
            processed_data['attention_mask'] = encoded_dict['attention_mask'].squeeze(0)

            processed_data['label'] = torch.LongTensor([int(data['label'])])

            self.processed_dataset.append(processed_data)

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self,
                    idx: int = None):
        return self.processed_dataset[idx]