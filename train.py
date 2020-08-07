from absl import app, flags, logging

import torch
from transformers import BertTokenizer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from model import BertClassificationModel, KoBertClassficationModel


FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', default=None,
                    help='Path to the training dataset')
flags.DEFINE_string('model', default=None,
                    help='Model to train (BERT, KoBERT)')
flags.DEFINE_string('save_dir', default=None,
                    help='Path to save model')
flags.DEFINE_string('version', default=None,
                    help='Explain experiment version')
flags.DEFINE_integer('cuda_device', default=0,
                     help='If given, uses this CUDA device in training')
flags.DEFINE_integer('max_epochs', default=10,
                     help='If given, uses this max epochs in training')
flags.DEFINE_integer('batch_size', default=4,
                     help='If given, uses this batch size in training')
flags.DEFINE_integer('num_workers', default=0,
                     help='If given, uses this number of workers in data loading')
flags.DEFINE_float('lr', default=2e-5,
                   help='If given, uses this learning rate in training')
flags.DEFINE_float('weight_decay', default=0.1,
                   help='If given, uses this weight decay in training')
flags.DEFINE_integer('warm_up', default=500,
                     help='If given, uses this warm up in training')


def main(argv):
    if FLAGS.model == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertClassificationModel(input_path=FLAGS.input_path,
                                        model='bert-base-multilingual-cased',
                                        tokenizer=tokenizer,
                                        batch_size=FLAGS.batch_size,
                                        num_workers=FLAGS.num_workers,
                                        lr=FLAGS.lr,
                                        weight_decay=FLAGS.weight_decay,
                                        warm_up=FLAGS.warm_up)
    elif FLAGS.model == 'KoBERT':
        bertmodel, vocab = get_pytorch_kobert_model()
        tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)
        model = KoBertClassficationModel(input_path=FLAGS.input_path,
                                         model=bertmodel,
                                         tokenizer=tokenizer,
                                         batch_size=FLAGS.batch_size,
                                         num_workers=FLAGS.num_workers,
                                         lr=FLAGS.lr,
                                         weight_decay=FLAGS.weight_decay,
                                         warm_up=FLAGS.warm_up)
    elif FLAGS.model == 'KcBERT':
        tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-large')
        model = BertClassificationModel(input_path=FLAGS.input_path,
                                        model='beomi/kcbert-large',
                                        tokenizer=tokenizer,
                                        batch_size=FLAGS.batch_size,
                                        num_workers=FLAGS.num_workers,
                                        lr=FLAGS.lr,
                                        weight_decay=FLAGS.weight_decay,
                                        warm_up=FLAGS.warm_up)
    else:
        raise ValueError('Unknown model type')

    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(
        filepath=FLAGS.save_dir,
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        strict=False,
        verbose=False,
        mode='min'
    )

    logger = TensorBoardLogger(
        save_dir=FLAGS.save_dir,
        name='logs_' + FLAGS.model,
        version=FLAGS.version
    )
    lr_logger = LearningRateLogger()

    if FLAGS.cuda_device > 1:
        trainer = Trainer(deterministic=True,
                          gpus=FLAGS.cuda_device,
                          distributed_backend='ddp',
                          log_gpu_memory=True,
                          checkpoint_callback=checkpoint_callback,
                          early_stop_callback=early_stop,
                          max_epochs=FLAGS.max_epochs,
                          logger=logger,
                          callbacks=[lr_logger])
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info(f'Use the number of GPU: {FLAGS.cuda_device}')
    elif FLAGS.cuda_device == 1:
        trainer = Trainer(deterministic=True,
                          gpus=FLAGS.cuda_device,
                          log_gpu_memory=True,
                          checkpoint_callback=checkpoint_callback,
                          early_stop_callback=early_stop,
                          max_epochs=FLAGS.max_epochs,
                          logger=logger,
                          callbacks=[lr_logger])
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info(f'Use the number of GPU: {FLAGS.cuda_device}')
    else:
        trainer = Trainer(deterministic=True,
                          checkpoint_callback=checkpoint_callback,
                          early_stop_callback=early_stop,
                          max_epochs=FLAGS.max_epochs,
                          logger=logger,
                          callbacks=[lr_logger])
        logging.info('No GPU available, using the CPU instead.')
    trainer.fit(model)


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'input_path'
    ])
    app.run(main)