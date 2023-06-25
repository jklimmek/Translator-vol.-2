import argparse
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from .utils import read_file


class TranslatorDataset(Dataset):
    """
    Custom PyTorch dataset for a translation task.

    Args:
        de_tokens (list): List of tokenized German sequences.
        en_tokens (list): List of tokenized English sequences.
        max_len (int): Maximum length of the sequences.

    Attributes:
        max_len (int): Maximum length of the sequences.
        src (torch.Tensor): Tensor representing the source sequences.
        tgt (torch.Tensor): Tensor representing the target sequences.
    """

    def __init__(self, de_tokens, en_tokens, max_len=100):
        super().__init__()
        self.max_len = max_len
        self.src, self.tgt = self._build_dataset(de_tokens, en_tokens)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.tgt[index]

    def _build_dataset(self, de_tokens, en_tokens):
        """
        Builds the source and target tensors from the tokenized sequences.

        Args:
            de_tokens (list): List of tokenized German sequences.
            en_tokens (list): List of tokenized English sequences.

        Returns:
            tuple: A tuple containing the source and target tensors.
        """
        x = torch.zeros(len(de_tokens), self.max_len, dtype=torch.int64)
        y = torch.zeros(len(en_tokens), self.max_len + 1, dtype=torch.int64)
        for i, (src, tgt) in tqdm(enumerate(zip(de_tokens, en_tokens)), total=len(de_tokens), desc="Dataset", ncols=100):
            x[i, :len(src)] = torch.tensor(src)
            y[i, :len(tgt)] = torch.tensor(tgt)
        return x, y
    

def tokenize_files(
    de_file,
    en_file,
    de_tokenizer_file,
    en_tokenizer_file,
    max_len,
    min_len,
    unk_percentage,
    length_tolerance
):
    """
    Tokenizes the contents of German and English files using respective tokenizers.

    Args:
        de_file (str): Path to the German file.
        en_file (str): Path to the English file.
        de_tokenizer_file (str): Path to the German tokenizer file.
        en_tokenizer_file (str): Path to the English tokenizer file.
        max_len (int): Maximum length allowed for a sequence.
        min_len (int): Minimum length allowed for a sequence.
        unk_percentage (float): Maximum allowed percentage of unknown tokens in a sequence.
        length_tolerance (float): Maximum allowed length tolerance between German and English sequences.

    Returns:
        tuple: A tuple containing two lists of tokenized German and English sequences, respectively.
    """

    # Read German file.
    de_txt = read_file(de_file)

    # Load German tokenizer.
    de_tokenizer = Tokenizer.from_file(de_tokenizer_file)
    de_unk_id = de_tokenizer.token_to_id("[UNK]")
    de_tokens_list = []

    # Read English file.
    en_txt = read_file(en_file)

    # Load English tokenizer.
    en_tokenizer = Tokenizer.from_file(en_tokenizer_file)
    en_unk_id = en_tokenizer.token_to_id("[UNK]")
    en_tokens_list = []

    # Tokenize German and English lines.
    for de_line, en_line in tqdm(zip(de_txt, en_txt), total=len(de_txt), desc="Tokenizing", ncols=100):
        de_tokens = de_tokenizer.encode(de_line).ids
        en_tokens = en_tokenizer.encode(en_line).ids

        # Filtering based on sequence length, unknown token percentage, and length tolerance.
        if (
            min_len >= len(de_tokens)
            or len(de_tokens) >= max_len
            or min_len >= len(en_tokens)
            or len(en_tokens) >= max_len
        ):
            continue
        if (
            de_tokens.count(de_unk_id) / len(de_tokens) > unk_percentage
            or en_tokens.count(en_unk_id) / len(en_tokens) > unk_percentage
        ):
            continue
        if max(len(de_tokens), len(en_tokens)) / min(len(de_tokens), len(en_tokens)) > length_tolerance:
            continue

        # Append tokenized sequences to respective lists.
        de_tokens_list.append(de_tokens)
        en_tokens_list.append(en_tokens)

    return de_tokens_list, en_tokens_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dataset",
        description="Create datasets"
    )
    parser.add_argument("--de-file", type=str, help="German file.", required=True)
    parser.add_argument("--en-file", type=str, help="English file.", required=True)
    parser.add_argument("--de-tok", type=str, help="German tokenizer.", required=True)
    parser.add_argument("--en-tok", type=str, help="English tokenizer.", required=True)
    parser.add_argument("--out-file", type=str, help="Name of file to save train dataset.", required=True)
    parser.add_argument("--max-len", type=int, help="Max length of sentence.", default=100)
    parser.add_argument("--min-len", type=int, help="Min length of sentence.", default=10)
    parser.add_argument("--unk-percentage", type=float, help="Max percentage of [UNK] tokens in sentence.", default=0.15)
    parser.add_argument("--length-tolerance", type=float, help="Max diff in length between source and target.", default=2.0)
    args = parser.parse_args()

    # Tokenize files.
    de_tokens, en_tokens = tokenize_files(
        de_file=args.de_file,
        en_file=args.en_file,
        de_tokenizer_file=args.de_tok,
        en_tokenizer_file=args.en_tok,
        max_len=args.max_len,
        min_len=args.min_len,
        unk_percentage=args.unk_percentage,
        length_tolerance=args.length_tolerance
    )

    # Create dataset.
    dataset = TranslatorDataset(de_tokens, en_tokens, args.max_len)

    # Save dataset.
    torch.save(dataset, args.out_file)
