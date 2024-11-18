# pytorch-lightning module > BART Model

import argparse
from train import Subheading_Generation
from transformers.models.bart import BartForConditionalGeneration
import yaml
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_type", default='', type=str)
parser.add_argument("--hparams", default='~/logs/tb_logs/data_type/flag/version_0/hparams.yaml', type=str)
parser.add_argument("--model_binary", default='~/logs/model_chp/data_type/flag/last.ckpt', type=str)
args = parser.parse_args()

flag = '~'
os.makedirs(flag, exist_ok=True)

with open(args.hparams) as f:
    hparams = yaml.full_load(f)
    inf = Subheading_Generation.load_from_checkpoint(args.model_binary, hparams=hparams)
    inf.model.bart.save_pretrained(flag)