import argparse
import logging
import os
from statistics import mode
from xmlrpc.client import Boolean
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from dataset import Subheading_Dataset
from model import CSG

from transformers import ElectraForMaskedLM, ElectraForPreTraining, ElectraTokenizer, PreTrainedTokenizerFast, BartTokenizer, T5Tokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import time

parser = argparse.ArgumentParser(description='Token-based Discriminative Learning for Subheading Generation')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--data_type',
                            type=str,
                            default=f'yonhapnews', # Name of the dataset
                            help='data type')
        
        parser.add_argument('--train_file',
                            type=str,
                            default='train.csv', # File path of training dataset
                            help='train file')

        parser.add_argument('--test_file',
                            type=str,
                            default='test.csv', # File path for testing
                            help='test file')

        parser.add_argument('--bart_path',
                            type=str,
                            default='gogamza/kobart-base-v2', # facebook/bart-base 
                            help='bart pretrained path')
        
        parser.add_argument('--generator_path',
                            type=str,
                            default='monologg/koelectra-base-generator', # google/electra-base-generator 
                            help='electra generator pretrained path')
        
        parser.add_argument('--discriminator_path',
                            type=str,
                            default='monologg/koelectra-base-discriminator', 
                            help='electra discriminator pretrained path')
        
        parser.add_argument('--pooler_type',
                            type=str,
                            default='avg', 
                            help='bart encoder pooler type')

        parser.add_argument('--headline_col',
                            type=str,
                            default='headline', 
                            help='headline col name in dataset')
        
        parser.add_argument('--subheading_col',
                            type=str,
                            default='subheading', 
                            help='subheading col name in dataset')
        
        parser.add_argument('--body_col',
                            type=str,
                            default='body', 
                            help='text col name in dataset')

        parser.add_argument('--body_max_len',
                            type=int,
                            default=1024,
                            help='body max seq len')

        parser.add_argument('--subheading_max_len',
                            type=int,
                            default=95,
                            help='subtitle max seq len')

        parser.add_argument('--headline_max_len',
                            type=int,
                            default=30,
                            help='title max seq len')

        return parser

class Subheading_DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.headline_max_len = args.headline_max_len
        self.subheading_max_len = args.subheading_max_len
        self.body_max_len = args.body_max_len

        self.batch_size = args.batch_size

        self.data_type = args.data_type
        self.train_file_path = args.train_file
        self.test_file_path = args.test_file

        if args.generator_path is None:
            self.electra_tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-generator')
        else:
            self.electra_tokenizer = ElectraTokenizer.from_pretrained(args.generator_path) 

        if args.bart_path is None:
            self.bart_tokenizer = BartTokenizer.from_pretrained('gogamza/kobart-base-v2')
        else : 
            self.bart_tokenizer = BartTokenizer.from_pretrained(args.bart_path)

        self.num_workers = args.num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers', type=int, default=5,
                            help='num of worker for dataloader')
        return parser

    def setup(self, stage):
        # Dataset is being split
        self.train = Subheading_Dataset(self.args, os.path.join('dataset', args.data_type, self.train_file_path), self.bart_tokenizer, self.electra_tokenizer)
        self.test = Subheading_Dataset(self.args, os.path.join('dataset', args.data_type, self.test_file_path), self.bart_tokenizer ,self.electra_tokenizer)

    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.test,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
        return test

class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.save_hyperparameters(hparams)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # Specific arguments to model
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=8,
                            help='batch size for training (default: 96)')

        parser.add_argument('--mlm_probability',
                            type=float,
                            default=0.1,
                            help='mlm prob for electra generator inputs')

        parser.add_argument('--electra_weight',
                            type=float,
                            default=0.01,
                            help='electra discriminator loss weight')

        parser.add_argument('--lr',
                            type=float,
                            default=3e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        parser.add_argument('--model_path',
                            type=str,
                            default=None,
                            help='kobart model path')
       
        return parser

    def configure_optimizers(self):
        '''
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        '''
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, correct_bias=True)
        num_workers = self.hparams.num_workers
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')

        scheduler = get_cosine_schedule_with_warmup( optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'monitor': 'loss', 'interval': 'step', 'frequency': 1}
        return [optimizer], [lr_scheduler]

class Subheading_Generation(Base):
    def __init__(self, hparams, dm):
        super(Subheading_Generation, self).__init__(hparams)
        self.validation_step_outputs = []
        self.model = CSG(hparams)
        self.model.train()
        self.dm = dm  # Instance of Datamodule

    
    def train_dataloader(self):
        return self.dm.train_dataloader()

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        batch_loss = self(batch)
        self.log('train_loss', batch_loss, prog_bar=True)
        return (batch_loss)
    
    def training_step_end(self, training_step_outputs):
        step_loss_list = []
        for device_loss in training_step_outputs:
            step_loss_list.append(device_loss)
        total_step_loss = torch.stack(step_loss_list).mean()
        self.log('train_step_loss', total_step_loss, prog_bar=True)
        return total_step_loss
    
    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.validation_step_outputs.append(loss)
        return (loss)
    
    def validation_step_end(self, validation_step_outputs):
        step_loss_list = []
        for device_loss in validation_step_outputs:
            step_loss_list.append(device_loss)
        total_step_loss = torch.stack(step_loss_list).mean()
        return (total_step_loss)


    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        losses = []
        for loss in self.validation_step_outputs:
            losses.append(loss)
        avg_val_loss = torch.stack(losses).mean()
        self.log('val_loss', avg_val_loss, prog_bar=True)
        self.log("validation_epoch_average", epoch_average)
        self.validation_step_outputs.clear()  # Clearing the memory
        

if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = Subheading_DataModule.add_model_specific_args(parser)
    #parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--max_epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--accelerator', type=str, default="gpu", help='Number of GPUs to use')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices to use, e.g., number of GPUs')
    parser.add_argument('--default_root_dir', type=str, default='logs', help='Directory for logs and checkpoints')
    args = parser.parse_args()
    logging.info(args)
    print(args)

    dm = Subheading_DataModule(args)
    model = Subheading_Generation(args, dm)

    # Setting
    flag = f'bsz={args.batch_size}-lr={args.lr}-mlm={args.mlm_probability}-el_weight={args.electra_weight}' 
                        
    model_chp = os.path.join(args.default_root_dir, 'model_chp', args.data_type, flag)
    os.makedirs(model_chp, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', dirpath=model_chp, filename='{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True, save_last=True, mode='min', save_top_k=2)
    earlystop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.015, patience=2, verbose=True, mode='min')

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs', args.data_type), name=flag)
    wandb_logger = pl_loggers.WandbLogger(project=f'Subheading_Generation_{args.data_type}', name=flag, id= f'{args.data_type}-{flag}-{time.strftime("%Y-%m-%d", time.localtime(time.time()))}')
    lr_logger = pl.callbacks.LearningRateMonitor()

 
    trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    accelerator=args.accelerator,  # Set device type (e.g., "gpu")
    devices=args.devices,          # Set the number of GPUs
    default_root_dir=args.default_root_dir,
    callbacks=[checkpoint_callback, earlystop_callback, lr_logger])

    trainer.fit(model, dm)
