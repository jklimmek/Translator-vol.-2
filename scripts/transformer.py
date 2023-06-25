import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, emb_dim, dropout=0.0):
        super().__init__()
        pe = torch.arange(emb_dim, dtype=torch.float32).repeat(seq_len, 1)
        for i in range(seq_len):
            pe[i, 0::2] = torch.sin(i / 1000**(pe[i, 0::2] / emb_dim))
            pe[i, 1::2] = torch.cos(i / 1000**(pe[i, 1::2] / emb_dim))
        
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropput = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.get_buffer("pe")[:, :x.size(1), :]
        x = self.dropput(x)
        return x


class Transformer(pl.LightningModule):
    def __init__(
        self, 
        seq_len,
        vocab_size, 
        embed_dim, 
        nhead,
        dim_feedforward,
        activation,
        num_layers, 
        dropout,
        label_smoothing,
        lr,
        betas,
        weight_decay,
        batch_size,
        max_epochs,
        epochs,
        accumulation_steps
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.lr = lr
        self.batch_size = batch_size
        self.betas = betas
        self.weight_decay = weight_decay
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.max_epochs = max_epochs
        self.accumulation_steps = accumulation_steps

        self.positional_encoding = PositionalEncoding(seq_len, embed_dim, dropout)
        self.src_embedding = nn.Embedding(vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            norm_first=True
        )
        self.ffnn = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt, tgt_mask=None):
        src = self.src_embedding(src) * self.embed_dim**0.5
        tgt = self.tgt_embedding(tgt) * self.embed_dim**0.5

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        out = self.ffnn(out)
        return out
    
    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask
    

    def _common_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        logits = self(src, tgt_input, tgt_mask=self._generate_square_subsequent_mask(tgt_input.size(1)).cuda())
        loss = F.cross_entropy(logits.permute(1,2,0), tgt_output, label_smoothing=self.label_smoothing)
        return loss
    

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss
    

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss


    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=2839302 * self.max_epochs // (self.batch_size * self.accumulation_steps), 
            eta_min=0
        ) 
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate"
            }
        }