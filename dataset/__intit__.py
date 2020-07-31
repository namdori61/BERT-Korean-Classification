import os, sys

from dataset.bert_dataset import BertDataset
from dataset.kobert_dataset import KoBertDataset

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

__all__ = ['BertDataset','KoBertDataset']