import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

import argparse
from tokenizers import Tokenizer
from .utils import load_model, translate, factory_translate, string_transforms, Color


def inference(model, translate_fn, de_tok, en_tok, maxlen, device):
    """Infer from model.

    Args:
        model (torch.nn.Module): Model.
        translate_fn (function): Translation function.
        de_tok (Tokenizer): German tokenizer.
        en_tok (Tokenizer): English tokenizer.
        maxlen (int): Maximum length of the input sequence.
        device (str): Device (CPU or CUDA).
    
    Returns:
        None
    """
    
    print(f"Enter {Color.YELLOW}/exit{Color.ENDC} to quit.")

    while True:
        # Get user input.
        text = input("Enter text: ")

        # Check for exit condition.
        if text == "/exit":
            break

        # Translate.
        output = translate_fn(text, model, de_tok, en_tok, maxlen, device)

        # Prettify output.
        output = string_transforms(output)

        # Print output.
        print(f"{Color.GREEN}Output: {Color.ENDC}{output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Infer",
        description="Inference"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path model checkpoint file.")
    parser.add_argument("--hyperparams", type=str, default=None, help="Path to hyperparams YAML file.")
    parser.add_argument("--de-tok", type=str, required=True, help="Path to English tokenizer file.")
    parser.add_argument("--en-tok", type=str, required=True, help="Path to German tokenizer file.")
    parser.add_argument("--maxlen", type=int, default=100, help="Maximum length of the input sequence.")
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

    # Infer.
    inference(model, translate_fn, de_tok, en_tok, args.maxlen, device)