import torch
import numpy as np
import random
import string
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from typing import Optional, Union, List
import logging
from transformers import PreTrainedModel, PreTrainedTokenizer
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK datasets if not already present."""
    try:
        resources = ['punkt', 'wordnet', 'averaged_perceptron_tagger']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}') if resource == 'punkt' else nltk.data.find(f'corpora/{resource}')
            except LookupError:
                nltk.download(resource)
    except Exception as e:
        logger.warning(f"Error downloading NLTK data: {str(e)}")

# Download NLTK data at module initialization
download_nltk_data()

class TextPerturbation:
    """Class for applying various text perturbation techniques."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the TextPerturbation class.
        
        Args:
            seed (int, optional): Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    @staticmethod
    def _get_word_substitutions(word: str) -> List[str]:
        """Get possible word substitutions using WordNet."""
        substitutions = set()
        synsets = wordnet.synsets(word)
        
        for syn in synsets:
            for lemma in syn.lemmas():
                if lemma.name() != word and "_" not in lemma.name():
                    substitutions.add(lemma.name())
        
        return list(substitutions)
    
    def apply_char_swap(self, text: str, prob: float = 0.1) -> str:
        """Randomly swap adjacent characters in words.
        
        Args:
            text (str): Input text
            prob (float): Probability of swapping characters in each word
            
        Returns:
            str: Perturbed text
        """
        words = text.split()
        perturbed_words = []
        
        for word in words:
            if len(word) > 1 and random.random() < prob:
                char_list = list(word)
                idx = random.randint(0, len(char_list)-2)
                char_list[idx], char_list[idx+1] = char_list[idx+1], char_list[idx]
                perturbed_words.append(''.join(char_list))
            else:
                perturbed_words.append(word)
                
        return ' '.join(perturbed_words)
    
    def apply_typo(self, text: str, prob: float = 0.1) -> str:
        """Introduce random typographical errors.
        
        Args:
            text (str): Input text
            prob (float): Probability of introducing typo in each word
            
        Returns:
            str: Text with introduced typos
        """
        words = text.split()
        qwerty_neighbors = {
            'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'srfce', 'e': 'wrsdf',
            'f': 'dcvgt', 'g': 'fvbht', 'h': 'gbnjy', 'i': 'ujko', 'j': 'huknm',
            'k': 'jlmi', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
            'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awedxz', 't': 'rfgy',
            'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
            'z': 'asx'
        }
        
        for i, word in enumerate(words):
            if len(word) > 1 and random.random() < prob:
                char_pos = random.randint(0, len(word)-1)
                char = word[char_pos].lower()
                if char in qwerty_neighbors:
                    typo_char = random.choice(qwerty_neighbors[char])
                    words[i] = word[:char_pos] + typo_char + word[char_pos+1:]
                    
        return ' '.join(words)
    
    def apply_word_drop(self, text: str, prob: float = 0.1) -> str:
        """Randomly drop words from the text.
        
        Args:
            text (str): Input text
            prob (float): Probability of dropping each word
            
        Returns:
            str: Text with dropped words
        """
        words = text.split()
        return ' '.join([word for word in words if random.random() > prob])
    
    def apply_word_substitute(self, text: str, prob: float = 0.1) -> str:
        """Replace words with their synonyms.
        
        Args:
            text (str): Input text
            prob (float): Probability of replacing each word
            
        Returns:
            str: Text with word substitutions
        """
        words = text.split()
        for i, word in enumerate(words):
            if random.random() < prob:
                substitutions = self._get_word_substitutions(word)
                if substitutions:
                    words[i] = random.choice(substitutions)
        return ' '.join(words)
    
    def apply_back_translation(self, 
                             text: str, 
                             model: PreTrainedModel,
                             tokenizer: PreTrainedTokenizer,
                             intermediate_lang: str = "fr",
                             max_length: int = 512) -> str:
        """Apply back-translation perturbation using a translation model.
        
        Args:
            text (str): Input text
            model: Translation model
            tokenizer: Model tokenizer
            intermediate_lang (str): Intermediate language code
            max_length (int): Maximum sequence length
            
        Returns:
            str: Back-translated text
        """
        try:
            # Translate to intermediate language
            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            with torch.no_grad():
                translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[intermediate_lang])
            intermediate = tokenizer.decode(translated[0], skip_special_tokens=True)
            
            # Translate back to English
            inputs = tokenizer(intermediate, return_tensors="pt", max_length=max_length, truncation=True)
            with torch.no_grad():
                back_translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["en"])
            return tokenizer.decode(back_translated[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Back-translation error: {str(e)}")
            return text
    
    def apply_adversarial(self, 
                         text: str,
                         model: PreTrainedModel,
                         tokenizer: PreTrainedTokenizer,
                         epsilon: float = 0.1,
                         device: str = 'cuda') -> str:
        """Apply gradient-based adversarial perturbation.
        
        Args:
            text (str): Input text
            model: Model to attack
            tokenizer: Model tokenizer
            epsilon (float): Perturbation magnitude
            device (str): Device to run model on
            
        Returns:
            str: Adversarially perturbed text
        """
        try:
            model.eval()
            model = model.to(device)
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            
            # Get embedding layer
            embeddings = model.get_input_embeddings()
            
            # Get initial embeddings
            embed_input = embeddings(input_ids)
            embed_input.requires_grad_()
            
            # Forward pass
            outputs = model(inputs_embeds=embed_input)
            
            # Compute loss (maximize likelihood of incorrect tokens)
            if hasattr(outputs, "logits"):
                loss = -outputs.logits.mean()
            else:
                loss = -outputs[0].mean()
                
            # Compute gradients
            loss.backward()
            
            # Generate perturbation
            perturb = epsilon * embed_input.grad.sign()
            
            # Apply perturbation
            perturbed_embed = embed_input + perturb
            
            # Generate from perturbed embedding
            with torch.no_grad():
                outputs = model.generate(
                    inputs_embeds=perturbed_embed,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Adversarial perturbation error: {str(e)}")
            return text
    
    def apply_mixed_perturbation(self, 
                               text: str,
                               prob: float = 0.1,
                               model: Optional[PreTrainedModel] = None,
                               tokenizer: Optional[PreTrainedTokenizer] = None) -> str:
        """Apply a random mix of different perturbation techniques.
        
        Args:
            text (str): Input text
            prob (float): Base probability for perturbations
            model: Optional model for advanced perturbations
            tokenizer: Optional tokenizer for advanced perturbations
            
        Returns:
            str: Perturbed text
        """
        perturbation_funcs = [
            (self.apply_char_swap, 0.3),
            (self.apply_typo, 0.3),
            (self.apply_word_drop, 0.2),
            (self.apply_word_substitute, 0.2)
        ]
        
        # Add model-based perturbations if model is provided
        if model is not None and tokenizer is not None:
            perturbation_funcs.extend([
                (lambda x: self.apply_adversarial(x, model, tokenizer), 0.1),
                (lambda x: self.apply_back_translation(x, model, tokenizer), 0.1)
            ])
        
        # Apply random perturbations
        perturbed_text = text
        for func, weight in perturbation_funcs:
            if random.random() < prob * weight:
                perturbed_text = func(perturbed_text)
                
        return perturbed_text

def apply_perturbation(
    text: str,
    perturbation_type: str,
    perturbation_level: float,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    device: str = 'cuda',
    seed: Optional[int] = None
) -> str:
    """Main function to apply perturbations to text.
    
    Args:
        text (str): Input text
        perturbation_type (str): Type of perturbation to apply
        perturbation_level (float): Level/probability of perturbation
        model: Optional model for advanced perturbations
        tokenizer: Optional tokenizer for advanced perturbations
        device (str): Device to use for model-based perturbations
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        str: Perturbed text
    """
    if perturbation_level <= 0:
        return text
        
    perturber = TextPerturbation(seed=seed)
    
    try:
        if perturbation_type == "typo":
            return perturber.apply_typo(text, perturbation_level)
        elif perturbation_type == "swap":
            return perturber.apply_char_swap(text, perturbation_level)
        elif perturbation_type == "drop":
            return perturber.apply_word_drop(text, perturbation_level)
        elif perturbation_type == "substitute":
            return perturber.apply_word_substitute(text, perturbation_level)
        elif perturbation_type == "adversarial":
            if model is None or tokenizer is None:
                logger.warning("Model and tokenizer required for adversarial perturbation")
                return text
            return perturber.apply_adversarial(text, model, tokenizer, perturbation_level, device)
        elif perturbation_type == "backtranslate":
            if model is None or tokenizer is None:
                logger.warning("Model and tokenizer required for back-translation")
                return text
            return perturber.apply_back_translation(text, model, tokenizer)
        elif perturbation_type == "mixed":
            return perturber.apply_mixed_perturbation(text, perturbation_level, model, tokenizer)
        else:
            logger.warning(f"Unknown perturbation type: {perturbation_type}")
            return text
            
    except Exception as e:
        logger.error(f"Error applying perturbation: {str(e)}")
        return text

# Example usage
if __name__ == "__main__":
    # Test text
    text = "This is a sample text to demonstrate various perturbation techniques."
    
    # Initialize perturbation class
    perturber = TextPerturbation(seed=42)
    
    # Test different perturbations
    print("Original:", text)
    print("Typo:", perturber.apply_typo(text, 0.3))
    print("Char swap:", perturber.apply_char_swap(text, 0.3))
    print("Word drop:", perturber.apply_word_drop(text, 0.2))
    print("Word substitute:", perturber.apply_word_substitute(text, 0.3))
