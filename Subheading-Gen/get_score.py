
from cmath import nan
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerFast, T5Tokenizer
from scorer import Score_Calculator
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_type", default='yonhap_news', type=str, help="Dataset type")
parser.add_argument("--headline_col", default='headline', type=str, help="Column name for headline")
parser.add_argument("--subheading_col", default='subheading', type=str, help="Column name for subheading")
parser.add_argument("--body_col", default='body', type=str, help="Column name for body text")
parser.add_argument("--device", default='cuda', type=str, help="Device to use (cuda or cpu)")
args = parser.parse_args()

# Language determination
language = 'kor' if args.data_type in ['yonhapnews', 'xlsum_kor'] else 'eng'

# Load tokenizer
tokenizer_name = 'gogamza/kobart-base-v2' if language == 'kor' else 'facebook/bart-base'
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)

# Dataset load
infer_data_path = os.path.join('dataset', args.data_type, 'infer_done.csv')
if not os.path.exists(infer_data_path):
    raise FileNotFoundError(f"{infer_data_path} not found. Please provide the correct path.")
    
infer_data = pd.read_csv(infer_data_path)
infer_data.reset_index(drop=True, inplace=True)

# Score columns
score_list = [
    'ref_gen_r1', 'ref_gen_r2', 'ref_gen_rl', 'ref_gen_bertscore',
    'gen_body_bleu_list', 'gen_body_bleu_max', 'gen_body_bleu_avg',
    'gen_body_r1_list', 'gen_body_r2_list', 'gen_body_rl_list',
    'gen_body_r1_max', 'gen_body_r1_avg', 'gen_body_r2_max',
    'gen_body_r2_avg', 'gen_body_rl_max', 'gen_body_rl_avg'
]
for score in score_list:
    infer_data[score] = None

# Initialize scorer
scorer = Score_Calculator(tokenizer=tokenizer, lang=language, device=args.device)
nan_list = []

# Score computation in rows 
for i in tqdm(range(len(infer_data))):
    try:
        instance = infer_data.iloc[i]
        headline = instance[args.headline_col]
        subheading = instance[args.subheading_col]
        body = instance[args.body_col]
        generated_subheading = instance[f'generated_{args.subheading_col}']
        
        # Score computation
        result_dict = scorer.compute(headline, subheading, body, generated_subheading)
        ref_gen = result_dict['ref_gen']
        gen_body_bleu = result_dict['gen_body_bleu']
        gen_body_rouge = result_dict['gen_body_rouge']

        
        infer_data.at[i, 'ref_gen_r1'] = ref_gen['r1']
        infer_data.at[i, 'ref_gen_r2'] = ref_gen['r2']
        infer_data.at[i, 'ref_gen_rl'] = ref_gen['rl']
        infer_data.at[i, 'ref_gen_bertscore'] = ref_gen['bert_score']
        infer_data.at[i, 'gen_body_bleu_list'] = gen_body_bleu['bleu_list']
        infer_data.at[i, 'gen_body_bleu_max'] = gen_body_bleu['bleu_max']
        infer_data.at[i, 'gen_body_bleu_avg'] = gen_body_bleu['bleu_avg']
        infer_data.at[i, 'gen_body_r1_list'] = gen_body_rouge['r1_list']
        infer_data.at[i, 'gen_body_r2_list'] = gen_body_rouge['r2_list']
        infer_data.at[i, 'gen_body_rl_list'] = gen_body_rouge['rl_list']
        infer_data.at[i, 'gen_body_r1_max'] = gen_body_rouge['r1_max']
        infer_data.at[i, 'gen_body_r1_avg'] = gen_body_rouge['r1_avg']
        infer_data.at[i, 'gen_body_r2_max'] = gen_body_rouge['r2_max']
        infer_data.at[i, 'gen_body_r2_avg'] = gen_body_rouge['r2_avg']
        infer_data.at[i, 'gen_body_rl_max'] = gen_body_rouge['rl_max']
        infer_data.at[i, 'gen_body_rl_avg'] = gen_body_rouge['rl_avg']

    except Exception as e:
        print(f"Error in row {i}: {e}")
        nan_list.append(i)

# Print NaN list
print('nan_list - ', nan_list)

# Update the dataset and save
output_path = os.path.join('dataset', args.data_type, 'infer_done', 'score_calculated.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
infer_data.to_csv(output_path, index=False)
print(f"Scores saved to {output_path}")
