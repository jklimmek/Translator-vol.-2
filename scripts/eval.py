import logging
import warnings
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

import argparse
import numpy as np
from tqdm import tqdm
from numpy import random
from sacrebleu.metrics import BLEU
from rouge import Rouge
from tokenizers import Tokenizer
from tabulate import tabulate
from .utils import Color, load_model, read_file, translate, factory_translate, string_transforms
    

def calculate_bleu(hyps, refs):
    """Calculate BLEU score.
    
    Args:
        hyps (list): List of hypotheses.
        refs (list): List of references.

    Returns:
        bleu (float): BLEU score.
    """
    bleu = BLEU()
    bleu_score = bleu.corpus_score(hyps, refs).score
    return round(bleu_score, 2)


def calculate_perplexity(probs):
    """Calculate perplexity.

    Args:
        probs (list): List of probabilities.

    Returns:
        perplexity (float): Perplexity.
    """
    perplexity = sum(np.prod(p) ** (-1 / len(p)) for p in probs) / len(probs)
    return round(perplexity, 2)


def calculate_rouge(hyps, refs):
    """Calculate ROUGE score.
    
    Args:
        hyps (list): List of hypotheses.
        refs (list): List of references.

    Returns:
        rouge (float): ROUGE score.
    """
    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps, refs, avg=True)
    return  round(rouge_score["rouge-1"]["f"], 2), \
            round(rouge_score["rouge-2"]["f"], 2), \
            round(rouge_score["rouge-l"]["f"], 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Eval",
        description="Evaluate model"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path model checkpoint file.")
    parser.add_argument("--hyperparams", type=str, default=None, help="Path to hyperparams YAML file.")
    parser.add_argument("--de-tok", type=str, required=True, help="Path to English tokenizer file.")
    parser.add_argument("--en-tok", type=str, required=True, help="Path to German tokenizer file.")
    parser.add_argument("--de-file", type=str, required=True, help="Path to English file.")
    parser.add_argument("--en-file", type=str, required=True, help="Path to German file.")
    parser.add_argument("--maxlen", type=int, default=100, help="Maximum length of the input sequence.")
    parser.add_argument("--size", type=int, default=100, help="Sample size.")
    parser.add_argument("--seed", type=int, default=2, help="Random seed.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA.")
    args = parser.parse_args()

    # Set device.
    device = "cuda" if args.cuda else "cpu"

    # Load model.
    model = load_model(args.checkpoint, args.hyperparams, device)

    # Choose proper translation function.
    translate_fn = factory_translate if args.hyperparams is None else translate

    # Load tokenizers.
    de_tok = Tokenizer.from_file(args.de_tok)
    en_tok = Tokenizer.from_file(args.en_tok)

    # Load files.
    de = read_file(args.de_file)
    refs = read_file(args.en_file)

    # Sample.
    random.seed(args.seed)
    indexes = random.randint(0, len(de), args.size)
    de = [de[i] for i in indexes]
    refs = [refs[i] for i in indexes]

    # Translate.
    hyps, probs = [], []
    for line in tqdm(de, desc="Translating", total=len(de), ncols=100):
        hyp, prob = translate_fn(line, model, de_tok, en_tok, args.maxlen, True, device)
        hyps.append(string_transforms(hyp))
        probs.append(prob)

    refs_list = [[e] for e in refs]

    # Calculate BLEU score.
    bleu = calculate_bleu(hyps, refs_list)

    # Calculate perplexity.
    perplexity = calculate_perplexity(probs)

    # Calculate ROUGE score.
    rouge_1, rouge_2, rouge_l = calculate_rouge(hyps, refs)

    # Create table.
    table = [
        [f"{Color.BLUE}BLEU{Color.ENDC}", bleu], 
        [f"{Color.PURPLE}Perplex.{Color.ENDC}", perplexity],
        [f"{Color.ROUGE}ROUGE-1{Color.ENDC}", rouge_1], 
        [f"{Color.ROUGE}ROUGE-2{Color.ENDC}", rouge_2], 
        [f"{Color.ROUGE}ROUGE-L{Color.ENDC}", rouge_l]
    ]

    # Print scores.
    print("\n", tabulate(table, headers=["Metric", "Score"], floatfmt=".2f"))