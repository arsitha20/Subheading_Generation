# Headline Token based Discriminative Learning for Subheading Generation in News Article

This is the pytorch implementation of **Headline Token based Discriminative Learning for Subheading Generation in News Article**
# Overview

The news subheading summarizes an article's contents in several sentences to support the headline limited to solely conveying the main contents. So, it is necessary to generate compelling news subheadings in consideration of the structural characteristics of the news. In this paper, we propose a subheading generation model using topical headline information. We introduce a discriminative learning method that utilizes the prediction result of masked headline tokens. Experiments show that the proposed model is effective and outperforms the comparative models on three news datasets written in two languages. We also show that our model performs robustly on a small dataset and various masking ratios. Qualitative analysis and human evaluations also shows that the overall quality of generated subheadings improved over the comparative models.

Our code is based on the code of https://github.com/Lainshower/Subheading-Gen/. Please refer to their repository for more information.

## DATASET

You can download the YonhapNews Data(Korean Data) from the following link
> Train data: [train](https://yonhap-news-dataset.s3.ap-northeast-2.amazonaws.com/yonhapnews/train.csv)
> Valid data: [valid](https://yonhap-news-dataset.s3.ap-northeast-2.amazonaws.com/yonhapnews/test.csv)
> Test data: [test](https://yonhap-news-dataset.s3.ap-northeast-2.amazonaws.com/yonhapnews/infer.csv)

we created our own english dataset by dividing the existing infer.csv into smaller subsets for training, testing, and infer, and saves these subsets in a xlsum_eng_small directory

## SETUP

cd Subheading-Gen

!pip install -r requirements.txt

!pip install --upgrade pip setuptools wheel

!pip install tokenizers==0.10.3

!curl -sSf https://sh.rustup.rs -o rust_install.sh

!sh rust_install.sh -y

!source $HOME/.cargo/env

# Set environment path for Rust
import os
os.environ['PATH'] += ":/root/.cargo/bin"

!pip install tokenizers --prefer-binary

!pip install transformers==4.10.0

!pip install torch pandas scikit-learn tqdm konlpy kss matplotlib seaborn wandb nltk datasets bert-score

!sudo apt update

!sudo apt install build-essential

!pip install pytorch-lightning

!chmod +x train_main.sh

## IMPLEMENTATION

Note: made few changes to infer.py to make these commands run.

for korean dataset:

train model:

!python train.py --data_type yonhapnews --default_root_dir logs --max_epochs 3 --accelerator gpu --devices 1 --mlm_probability 0.3 --electra_weight 0.01 --lr 3e-5

evaluate model:

!python infer.py \
    --model_path "/content/Subheading-Gen/logs/model_chp/yonhapnews/bsz=8-lr=3e-05-mlm=0.3-el_weight=0.01/last.ckpt" \
    --tokenizer_path "gogamza/kobart-base-v2" \
    --data_type yonhapnews

for english dataset:

training command:

!python train.py \
    --data_type xlsum_eng_small \
    --headline_col headline \
    --subheading_col subheading \
    --body_col body \
    --default_root_dir logs \
    --max_epochs 3 \
    --accelerator gpu \
    --devices 1 \
    --mlm_probability 0.3 \
    --electra_weight 0.01 \
    --lr 3e-5 \
    --batch-size 4 \
    --bart_path "facebook/bart-base" \
    --generator_path "google/electra-small-generator" \
    --discriminator_path "google/electra-small-discriminator" \
    --body_max_len 512 \
    --train_file train.csv \
    --test_file test.csv

evaluate model:

!python infer.py \
    --model_path "/content/Subheading-Gen/logs/model_chp/xlsum_eng_small/bsz=4-lr=3e-05-mlm=0.3-el_weight=0.01/last.ckpt" \
    --tokenizer_path "facebook/bart-large" \
    --data_type xlsum_eng_small
