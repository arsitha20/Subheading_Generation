import argparse
from ast import arg
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraForMaskedLM, ElectraForPreTraining, BartForConditionalGeneration
from transformers import ElectraTokenizer

def get_bart(bart_name):
    bart = BartForConditionalGeneration.from_pretrained(bart_name)
    return bart

def get_electra_tokenizer(gen_name):
    tokenizer = ElectraTokenizer.from_pretrained(gen_name)
    return tokenizer

def get_electra_discriminator(dis_name):
    discriminator = ElectraForPreTraining.from_pretrained(dis_name)
    return discriminator

def get_electra_generator(gen_name):
    generator = ElectraForMaskedLM.from_pretrained(gen_name)
    return generator

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.encoder_last_hidden_state
        hidden_states = outputs.encoder_hidden_states

        if self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))

        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result

        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
            
        else:
            raise NotImplementedError

class CSG(nn.Module):
    def __init__(self, args):
        super(CSG, self).__init__()

        if type(args) == dict:
            args = argparse.Namespace(**args)

        self.bart = get_bart(args.bart_path)
        self.pooler = Pooler(args.pooler_type)
        
        self.electra_weight = args.electra_weight
        
        self.electra_tokenizer = get_electra_tokenizer(args.generator_path)
        self.electra_generator = get_electra_generator(args.generator_path)
        self.electra_discriminator = get_electra_discriminator(args.discriminator_path)

        # Freeze ELECTRA Generator Parameters
        for param in self.electra_generator.parameters():
            param.requires_grad = False
    
    def average_pooling(self, attention_mask, hidden_states):
        """
        Applies average pooling to the encoder hidden states.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden_states = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_hidden_states / sum_mask

    def forward(self, inputs):
        # BART Input Processing
        bart_input = inputs['bart']
        
        # ELECTRA Input Processing
        electra_input = inputs['electra']

        # Forward pass through BART for subheading generation
        bart_output = self.bart(
            input_ids=bart_input['encoder_input_ids'],
            attention_mask=bart_input['encoder_attention_mask'],
            decoder_input_ids=bart_input['decoder_input_ids'],
            decoder_attention_mask=bart_input['decoder_attention_mask'],
            labels=bart_input['decoder_labels'],
            return_dict=True
        )
        
        # Loss
        bart_mlm_loss = bart_output.loss

        # average pooling  to BART output
        bart_encoder_pooler_output = self.average_pooling(
            bart_input['encoder_attention_mask'],
            bart_output.encoder_last_hidden_state
        )
        bart_encoder_pooler_output = bart_encoder_pooler_output.unsqueeze(1)

        # predictions with ELECTRA generator
        with torch.no_grad():
            g_pred = self.electra_generator(
                electra_input['generator_input_ids'],
                attention_mask=electra_input['attention_mask']
            ).logits.argmax(-1)
            
            # First token to CLS
            g_pred[:, 0] = self.electra_tokenizer.cls_token_id
            
            # Mismatches 
            replaced = (g_pred != electra_input['labels']) * electra_input['attention_mask']
            e_inputs = g_pred * electra_input['attention_mask']

        # ELECTRA discriminator
        discriminator_output = self.electra_discriminator(
            e_inputs,
            attention_mask=electra_input['attention_mask'],
            labels=replaced.view(-1, replaced.size(-1)),
            return_dict=True
        )

        #Loss
        discriminator_loss = discriminator_output.loss

        loss = bart_mlm_loss + self.electra_weight * discriminator_loss

        return loss
