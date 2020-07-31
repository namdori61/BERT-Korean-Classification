import os, sys

from model.bert_model import BertClassficationModel

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

__all__ = ['BertClassficationModel']