import re
import yaml
import torch
import torch.nn.functional as F
from .transformer import Transformer
from .factory import FactoryModel


PARAM_MAPPING = {
    """
    A dictionary that maps original parameter names to new names.
    """
    "vocab_size": "vs",
    "embedding_dim": "emb",
    "sequence_length": "len",
    "num_layers": "n_layers",
    "num_heads": "n_heads",
    "hidden_layer_multiplier": "h_mult",
    "use_rotary_embeddings": "re",
    "dropout": "dp",
    "attention": "att",
    "activation": "act",
    "normalization": "norm",
    "weight_decay": "wd",
    "max_epochs": "max_ep",
    "betas": "betas",
    "learning_rate": "lr",
    "batch_size": "bs",
    "accumulation_steps": "acc",
    "warmup_period": "wp",
    "comment": "cmt",
}

class Color:
    """
    A class that defines color codes for printing colored text in the terminal.
    """
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    ROUGE = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'


def read_file(path):
    """
    Reads a file and returns its contents as a list of lines.

    Args:
        path (str): Path to the file.

    Returns:
        list: List of lines from the file.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def save_file(file, path):
    """
    Saves the contents of a file from a list of lines.

    Args:
        file (list): List of lines to be written to the file.
        path (str): Path to the file.
    """
    with open(path, "w") as f:
        for line in file:
            f.write(line + "\n")


def yaml_to_kwargs(filename):
    """
    Reads a YAML file and returns its values as keyword arguments (kwargs).

    Args:
        filename (str): The path to the YAML file.

    Returns:
        dict: A dictionary containing the YAML values as keyword arguments.
    """
    with open(filename, "r") as file:
        data = yaml.safe_load(file)

    kwargs = {}
    for key, value in data.items():
        kwargs[key] = value

    return kwargs


def create_kwarg_string(param_mapping=None, **kwargs):
    """
    Creates a string by concatenating keyword argument names and values.

    Args:
        param_mapping (dict, optional): A dictionary that maps original parameter names to new names.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        str: A string in the format "kwarg1-value1_kwarg2-value2...kwargN-valueN".
    """
    if param_mapping is not None:
        kwargs = {param_mapping.get(key, key): value for key, value in kwargs.items() 
                  if key in param_mapping and kwargs[key] != ""}
    kwarg_string = '__'.join([f"{key}-{value}" for key, value in kwargs.items()])
    return kwarg_string


def string_transforms(string):
    """Apply string transformations to bump up scores.

    Args:
        string (str): String to apply transformations.

    Returns:
        string (str): Transformed string.
    """
    string = re.sub(r" ' ", "'", string)
    string = re.sub(r" ([.,:;])", r"\1", string)
    string = re.sub(r" i ", r" I ", string)
    string = re.sub(r'\b([ap])\.\s*m\b', r'\1.m', string)
    string = re.sub(r'(\d+)\s+(st|nd|rd|th)\b', r'\1\2', string)
    string = re.sub(r"(\d+)( \d+)+", lambda match: match.group(0).replace(" ", ""), string)
    string = re.sub(r"(\d+.\s)+", lambda match: match.group(0).replace(" ", "") + " ", string)
    return string[:1].upper() + string[1:]


def load_model(checkpoint, hyperparams=None, device="cpu"):
    """
    Loads a model from a checkpoint.

    Args:
        checkpoint (str): Path to the checkpoint file.
        hyperparams (dict, optional): A dictionary containing the model hyperparameters. Defaults to None.
        device (str, optional): The device to load the model on. Defaults to "cpu".

    Returns:
        pl.LightningModule: The loaded model.
    """
    if hyperparams is None:
        model = FactoryModel.load_from_checkpoint(checkpoint).to(device)
        model.eval()
        return model
    
    params = yaml_to_kwargs(hyperparams)
    model = Transformer.load_from_checkpoint(
        checkpoint, 
        **params["model_hyperparams"], 
        **params["model_training"]
    ).to(device)
    model.eval()
    return model


@torch.no_grad()
def factory_translate(sentence, model, de_tok, en_tok, maxlen=100, return_probs=False, device="cpu"):
    """
    Translates a sentence from a source language to a target language using a factory translation model.

    Args:
        sentence (str): The input sentence to be translated.
        model (torch.nn.Module): The translation model.
        de_tok (Tokenizer): The tokenizer for the source language.
        en_tok (Tokenizer): The tokenizer for the target language.
        maxlen (int, optional): The maximum length of the translated sentence. Defaults to 100.
        return_probs (bool, optional): Whether to return word probabilities or not. Defaults to False.
        device (str, optional): The device to run the translation on. Defaults to "cpu".

    Returns:
        str: The translated sentence.
        list: The word probabilities.
    """
    model.eval()
    de_tok.enable_padding(length=maxlen)
    de_tok.enable_truncation(max_length=maxlen)
    x = torch.tensor(de_tok.encode(sentence, add_special_tokens=False).ids, device=device).unsqueeze(0)
    y = torch.zeros((1, maxlen), dtype=torch.long, device=device)
    y[0, 0] = en_tok.token_to_id("[SOS]")
    probs = []
    for i in range(1, maxlen):
        logits = model(src=x, tgt=y)
        prob, token = F.softmax(logits[0, i-1], dim=-1).topk(1)
        y[0, i] = token.item()
        probs.append(prob.item())
        if token == en_tok.token_to_id("[EOS]"):
            break

    translated = en_tok.decode(y.tolist()[0])
    if return_probs is True:
        return translated, probs
    return translated


@torch.no_grad()
def translate(sentence, model, de_tok, en_tok, maxlen=100, return_probs=False, device="cpu"):
    """
    Translates a sentence from a source language to a target language using a translation model.

    Args:
        sentence (str): The input sentence to be translated.
        model (torch.nn.Module): The translation model.
        de_tok (Tokenizer): The tokenizer for the source language.
        en_tok (Tokenizer): The tokenizer for the target language.
        maxlen (int, optional): The maximum length of the translated sentence. Defaults to 100.
        return_probs (bool, optional): Whether to return the probabilities of the tokens. Defaults to False.
        device (str, optional): The device to run the translation on. Defaults to "cpu".

    Returns:
        str: The translated sentence.
        list: The probabilities of the tokens.
    """
    model.eval()
    de_tok.enable_padding(length=maxlen)
    de_tok.enable_truncation(max_length=maxlen)
    x = torch.tensor(de_tok.encode(sentence, add_special_tokens=False).ids, device=device) 
    y = torch.tensor([[en_tok.token_to_id("[SOS]")]], dtype=torch.long, device=device)
    probs = []
    while y.size(1) < maxlen:
        tgt_mask = model._generate_square_subsequent_mask(y.size(1)).to(device)
        logits = model(x, y, tgt_mask=tgt_mask)
        prob, token = F.softmax(logits, dim=-1).topk(1)
        probs.append(prob[-1].item())
        token = torch.tensor([[token[-1].item()]], device=device)
        y = torch.cat((y, token), dim=1)
        if token == en_tok.token_to_id("[EOS]"):
            break
        
    translated = en_tok.decode(y.tolist()[0])
    if return_probs is True:
        return translated, probs
    return translated
