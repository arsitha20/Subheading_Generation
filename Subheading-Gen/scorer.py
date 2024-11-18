
import nltk
nltk.download('punkt_tab') 

import evaluate
from nltk.tokenize import sent_tokenize
from metrics.rouge import Rouge
import numpy as np

class Score_Calculator:
    def __init__(self, tokenizer, sent_tokenizer=sent_tokenize, lang='kor', smooth=False, epsilon=1e-7, device='cuda:0'):
        self.tokenizer = tokenizer
        self.sent_tokenizer = sent_tokenizer
        self.lang = lang
        self.smooth = smooth 
        self.epsilon = epsilon  
        self.bert_scorer = evaluate.load("bertscore")  
        self.bleu_scorer = evaluate.load("bleu")  
        self.rouge_scorer = Rouge(rouge_types=["rouge1", "rouge2", "rougeL"])
        self.device = device

    def wrap_bleu_calculate(self, reference_token, prediction_token_list):
        bleu_list = []

        for sent_token in prediction_token_list:
            bleu = self.bleu_scorer.compute(
                predictions=[sent_token],
                references=[[reference_token]],
                smooth=self.smooth
            )['bleu']
            bleu_list.append(bleu)

        assert len(prediction_token_list) == len(bleu_list), "BLEU Scorer Calculation Error"

        bleu_max, bleu_avg = np.max(bleu_list), np.mean(bleu_list)
        return bleu_list, bleu_max, bleu_avg

    
    def rouge_calculate(self, reference, prediction):
        reference = " ".join(reference) if isinstance(reference, list) else reference
        prediction = " ".join(prediction) if isinstance(prediction, list) else prediction
        rouge_score_dict = self.rouge_scorer.score(reference, prediction)
        return (
            rouge_score_dict['rouge1']['f1_score'],
            rouge_score_dict['rouge2']['f1_score'],
            rouge_score_dict['rougeL']['f1_score']
        )

    def wrap_rouge_calculate(self, reference_token, prediction_token_list):
        r1_list, r2_list, rl_list = [], [], []

        for sent_token in prediction_token_list:
            r1, r2, rl = self.rouge_calculate(reference_token, sent_token)
            r1_list.append(r1)
            r2_list.append(r2)
            rl_list.append(rl)

        assert len(r1_list) == len(r2_list) == len(rl_list) == len(prediction_token_list), "Rouge Scorer Calculation Error"

        r1_max, r1_avg = np.max(r1_list), np.mean(r1_list)
        r2_max, r2_avg = np.max(r2_list), np.mean(r2_list)
        rl_max, rl_avg = np.max(rl_list), np.mean(rl_list)

        return r1_list, r2_list, rl_list, r1_max, r1_avg, r2_max, r2_avg, rl_max, rl_avg

   

    def compute(self, title, summary, text, generated_summary):
        """
        Compute metrics for generated summary against reference summary and body text.

        Args:
            title (str): Title of the text.
            summary (str): Reference subheading (summary).
            text (str): Body of the article.
            generated_summary (str): Generated subheading.

        Returns:
            dict: Dictionary containing ROUGE, BLEU, and BERTScore metrics.
        """
        # Input
        print(f"Title: {title}")
        print(f"Summary: {summary}")
        print(f"Text: {text[:100]}...")  
        print(f"Generated Summary: {generated_summary}")

        # Skipping the invalid inputs 
        if not isinstance(title, str) or not isinstance(summary, str) or not isinstance(text, str) or not isinstance(generated_summary, str):
            print("Invalid input detected. Skipping...")
            return None

        try:
          
            reference_summary = summary if isinstance(summary, str) else " ".join(summary)
            generated_summary_str = generated_summary if isinstance(generated_summary, str) else " ".join(generated_summary)
            body_sentences = self.sent_tokenizer(text)

            # ROUGE Scores b/w references
            ref_gen_r1, ref_gen_r2, ref_gen_rl = self.rouge_calculate(reference_summary, generated_summary_str)

            # BERTScore
            ref_gen_bs = self.bert_scorer.compute(
                predictions=[generated_summary_str],
                references=[reference_summary],
                lang=self.lang
            )['f1'][0]

            # ROUGE AND BLEU scores
            r1_text_list, r2_text_list, rl_text_list, r1_max, r1_avg, r2_max, r2_avg, rl_max, rl_avg = self.wrap_rouge_calculate(
                generated_summary_str, body_sentences
            )
            bleu_text_list, bleu_max, bleu_avg = self.wrap_bleu_calculate(
                generated_summary_str, body_sentences
            )

            # Metrics
            print(f"ROUGE: {ref_gen_r1}, {ref_gen_r2}, {ref_gen_rl}")
            print(f"BERTScore: {ref_gen_bs}")
            print(f"BLEU Avg: {bleu_avg}, BLEU Max: {bleu_max}")

            # Result
            result_dict = dict(
                ref_gen=dict(r1=ref_gen_r1, r2=ref_gen_r2, rl=ref_gen_rl, bert_score=ref_gen_bs),
                gen_text_bleu=dict(bleu_list=bleu_text_list, bleu_max=bleu_max, bleu_avg=bleu_avg),
                gen_text_rouge=dict(
                    r1_list=r1_text_list, r2_list=r2_text_list, rl_list=rl_text_list,
                    r1_max=r1_max, r1_avg=r1_avg, r2_max=r2_max, r2_avg=r2_avg, rl_max=rl_max, rl_avg=rl_avg
                )
            )

            return result_dict

        except Exception as e:
            print(f"Error calculating metrics for title '{title}': {e}")
            return None
