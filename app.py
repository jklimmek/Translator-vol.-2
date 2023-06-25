import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

import torch
from tokenizers import Tokenizer
from scripts.utils import load_model, factory_translate, string_transforms
from flask import Flask, render_template, request


app = Flask(__name__)


device = "cuda" if torch.cuda.is_available() else "cpu"
de_tok = Tokenizer.from_file("tokenizers/de_tokenizer_10000.json")
en_tok = Tokenizer.from_file("tokenizers/en_tokenizer_10000.json")
model = load_model("inference_models/epoch=04-train_loss=0.6309-val_loss=0.6159_fct_18m.ckpt", device=device)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        text = request.form["input-text"]
        try:
            translated_text = factory_translate(text, model, de_tok, en_tok, device=device)
            translated_text = string_transforms(translated_text)
        except RuntimeError:
            return "<script>alert('Max length cannot exceed 100 tokens.');</script>"
        return render_template("index.html", input_text=text, translated_text=translated_text)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)