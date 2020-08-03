import os, sys

from model.bert_model import BertClassificationModel
from model.kobert_model import KoBertClassficationModel

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

__all__ = ['BertClassificationModel','KoBertClassficationModel']