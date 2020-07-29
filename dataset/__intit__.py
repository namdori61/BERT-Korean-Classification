import os, sys

from dataset.bert_dataset import BertDataset

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

__all__ = ['BertDataset']