import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_type", default='yonhapnews', type=str, help="Dataset type")
parser.add_argument("--headline_col", default='headline', type=str, help="Column name for headline")
parser.add_argument("--subheading_col", default='subheading', type=str, help="Column name for subheading")
parser.add_argument("--body_col", default='body', type=str, help="Column name for body text")
parser.add_argument("--encoder_max_len", default=1024, type=int, help="Maximum input length for encoder")
parser.add_argument("--generate_max_len", default=95, type=int, help="Maximum output length")
parser.add_argument("--device", default='cuda', type=str, help="Device to use (cuda or cpu)")
parser.add_argument("--model_path", required=True, type=str, help="Path to the trained model folder or Hugging Face repo ID")
parser.add_argument("--tokenizer_path", required=True, type=str, help="Path to the tokenizer folder or Hugging Face repo ID")
args = parser.parse_args()

# model
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
model.to(args.device)

# Infer dataset
infer_data_path = os.path.join('dataset', args.data_type, 'infer.csv')
if not os.path.exists(infer_data_path):
    raise FileNotFoundError(f"{infer_data_path} not found. Check the correct path")

# Normalize the dataset
infer_data = pd.read_csv(infer_data_path)
infer_data.columns = infer_data.columns.str.strip().str.lower()  
print("Columns in the dataset:", infer_data.columns)

# Check col names exists
required_columns = [args.headline_col, args.subheading_col, args.body_col]
for col in required_columns:
    if col not in infer_data.columns:
        raise KeyError(f"Column missing: {col}")

infer_data[f'generated_{args.subheading_col}'] = np.nan

# Subheading generation
for i in tqdm(range(len(infer_data))):
    input_text = infer_data[args.body_col].iloc[i]
    if pd.isna(input_text) or not isinstance(input_text, str):
        print(f"Skipping row {i} due to invalid body text.")
        continue
    
    input_ids = tokenizer.encode(
        input_text,
        truncation=True,
        max_length=args.encoder_max_len,
        return_tensors="pt"
    ).to(args.device)
    
    output = model.generate(
        input_ids,
        max_length=args.generate_max_len,
        num_beams=5,
        no_repeat_ngram_size=5,
        eos_token_id=1
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    infer_data.loc[i, f'generated_{args.subheading_col}'] = generated_text

    # Debugging logs
    print("Headline:", infer_data[args.headline_col][i])
    print("Original Subheading:", infer_data[args.subheading_col][i])
    print("Generated Subheading:", generated_text)

# Save results
output_path = os.path.join('dataset', args.data_type, 'infer_done.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
infer_data.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
