import torch
import torch.nn as nn
import pytorch_lightning as pl
from xformers.factory.model_factory import xFormer, xFormerConfig


class FactoryModel(pl.LightningModule):
    def __init__(
            self, 
            vocab_size,
            embedding_dim,
            sequence_length,
            num_layers,
            num_heads,
            hidden_layer_multiplier,
            dropout,
            attention,
            activation,
            normalization,
            weight_decay,
            epochs,
            max_epochs,
            betas,
            learning_rate,
            batch_size,
            accumulation_steps,
            warmup_period,
    ):
        super().__init__()
        self.save_hyperparameters()

        config = [
            ##### Transformer Encoder #####
            {
                "block_type": "encoder",
                "reversible": False,
                "num_layers": self.hparams.num_layers,
                "dim_model": self.hparams.embedding_dim,
                "residual_norm_style": self.hparams.normalization,
                "position_encoding_config": {
                    "name": "vocab",
                    "seq_len": self.hparams.sequence_length,
                    "vocab_size": self.hparams.vocab_size,
                },
                "multi_head_config": {
                    "num_heads": self.hparams.num_heads,
                    "residual_dropout": self.hparams.dropout,
                    "use_rotary_embeddings": True,
                    "attention": {
                        "name": self.hparams.attention,
                        "dropout": self.hparams.dropout,
                        "causal": False,
                        "seq_len": self.hparams.sequence_length,
                    },
                },
                "feedforward_config": {
                    "name": "MLP",
                    "dropout": self.hparams.dropout,
                    "activation": self.hparams.activation,
                    "hidden_layer_multiplier": self.hparams.hidden_layer_multiplier,
                }
            },
            ##### Transformer Decoder #####
            {
                "block_type": "decoder",
                "reversible": False,
                "num_layers": self.hparams.num_layers,
                "dim_model": self.hparams.embedding_dim,
                "residual_norm_style": self.hparams.normalization,
                "position_encoding_config": {
                    "name": "vocab",
                    "seq_len": self.hparams.sequence_length,
                    "vocab_size": self.hparams.vocab_size,
                },
                "multi_head_config_masked": {
                    "num_heads": self.hparams.num_heads,
                    "residual_dropout": self.hparams.dropout,
                    "use_rotary_embeddings": True,
                    "attention": {
                        "name": "scaled_dot_product",
                        "dropout": self.hparams.dropout,
                        "causal": True,
                        "seq_len": self.hparams.sequence_length,
                    },
                },
                "multi_head_config_cross": {
                    "num_heads": self.hparams.num_heads,
                    "residual_dropout": self.hparams.dropout,
                    "attention": {
                        "name": "scaled_dot_product",
                        "dropout": self.hparams.dropout,
                        "causal": True,
                        "seq_len": self.hparams.sequence_length,
                    },
                },
                "feedforward_config": {
                    "name": "MLP",
                    "dropout": self.hparams.dropout,
                    "activation": self.hparams.activation,
                    "hidden_layer_multiplier": self.hparams.hidden_layer_multiplier,
                },
            }
        ]
        xformer_config = xFormerConfig(config)
        self.xformer = xFormer.from_config(xformer_config)
        self.layer_norm = nn.LayerNorm(self.hparams.embedding_dim)
        self.linear = nn.Linear(self.hparams.embedding_dim, self.hparams.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src, tgt):
        x = self.xformer(src, tgt)
        x = self.layer_norm(x)
        x = self.linear(x)
        return x
    
    def _common_step(self, batch):
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        logits = self(src, tgt_input)
        loss = self.loss_fn(logits.permute(0, 2, 1), tgt_output)
        return loss

    def training_step(self, batch, _):
        loss = self._common_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, _):
        loss = self._common_step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        decay = []
        no_decay = []
        no_decay_layers = ["bias", "norm"]

        for name, param in self.named_parameters():
            if any(nd in name for nd in no_decay_layers):
                no_decay.append(param)
            else:
                decay.append(param)
        
        optim_groups = [
            {"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
        )
        # Train: 2839302, Debug: 4558
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, 
                T_max=2839302 * self.hparams.max_epochs // (self.hparams.batch_size * self.hparams.accumulation_steps), 
                eta_min=0
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]