import argparse
from .utils import read_file
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)


def create_tokenizer(txt, vocab_size, add_sentinel_tokens=False, init_alphabet=[], limit_alphabet=None):
    """
    Creates and trains a tokenizer using the given text data.

    Args:
        txt (Iterable[str]): Iterable containing the text data to train the tokenizer.
        vocab_size (int): Size of the vocabulary for the tokenizer.
        add_sentinel_tokens (bool, optional): Whether to add start-of-sequence and end-of-sequence tokens. Defaults to False.
        init_alphabet (list, optional): List of characters to include in the initial alphabet. Defaults to an empty list.
        limit_alphabet (int, optional): Maximum number of characters to keep in the alphabet. Defaults to None.

    Returns:
        tokenizers.Tokenizer: Trained tokenizer object.
    """

    # Initialize the tokenizer with WordPiece model and set the unknown token.
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    # Apply normalization to lowercase the text.
    tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase()])

    # Pre-tokenization: separate digits and whitespace.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.Whitespace()
    ])

    # Define special tokens for the tokenizer.
    special_tokens = ["[PAD]", "[UNK]"]

    # Add start-of-sequence and end-of-sequence tokens if specified.
    if add_sentinel_tokens is True:
        special_tokens.extend(["[SOS]", "[EOS]"]) 

    # Configure the WordPiece trainer with vocabulary size, special tokens, and alphabet settings.
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=init_alphabet,
        limit_alphabet=limit_alphabet
    )

    # Set the WordPiece decoder with a prefix.
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    # Train the tokenizer on the given text data using the trainer.
    tokenizer.train_from_iterator(txt, trainer)

    # Return the trained tokenizer if no special tokens are specified.
    if add_sentinel_tokens is False:
        return tokenizer
    
    # Get the token IDs for the special tokens.
    sos_token = tokenizer.token_to_id("[SOS]")
    eos_token = tokenizer.token_to_id("[EOS]")

    # Configure the post-processor to handle [SOS] and [EOS] tokens.
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[SOS]:0 $A:0 [EOS]:0",
        special_tokens=[("[SOS]", sos_token), ("[EOS]", eos_token)],
    )

    # Return the trained tokenizer.
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Tokenizer",
        description="Creating tokenizer"
    )
    parser.add_argument("--file", type=str, help="Name of file.", required=True)
    parser.add_argument("--vocab-size", type=int, help="Desired vocab size.", required=True)
    parser.add_argument("--add-sentinel-tokens", action="store_true", help="Add SOS and EOS tokens.")
    parser.add_argument("--output-path", type=str, help="Path to save tokenizer.", required=True)
    parser.add_argument("--initial-alphabet", type=str, help="Initial alphabet.", default="abcdefghijklmnopqrstuvwxyz1234567890().,/:;?!'^#$%")
    parser.add_argument("--additional-alphabet", type=str, help="Additional alphabet.", default="")
    args = parser.parse_args()

    # Read file.
    txt = read_file(args.file)

    # Merge alphabets.
    init_alphabet = list(args.initial_alphabet)
    add_alphabet = list(args.additional_alphabet)
    init_alphabet.extend(add_alphabet)

    # Create tokenizer.
    tokenizer = create_tokenizer(
        txt=txt,
        vocab_size=args.vocab_size,
        add_sentinel_tokens=args.add_sentinel_tokens,
        init_alphabet=init_alphabet,
        limit_alphabet=len(init_alphabet)
    )

    # Save tokenizer.
    tokenizer.save(args.output_path)
