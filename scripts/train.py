import os
import argparse
import logging
import warnings
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .dataloader import TranslatorDataloader
from .dataset import TranslatorDataset
from .factory import FactoryModel
from .transformer import Transformer
from .utils import yaml_to_kwargs, create_kwarg_string, PARAM_MAPPING

# import tensorboard 

# TODO: find better way of calculating T_max in optimizer.


def train(hyperparams, config, comment=""):

    # Unpack hyperparams.
    model_hyperparams = hyperparams["model_hyperparams"]
    model_training = hyperparams["model_training"]

    # Load datasets.
    train_dataset = torch.load(config["train_file"])
    dev_dataset = torch.load(config["dev_file"])

    # Create dataloader.
    dataloader = TranslatorDataloader(train_dataset, dev_dataset, model_training["batch_size"], num_workers=6)

    # Monitor learning rate.
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Save model checkpoints.
    architecture_name = create_kwarg_string(param_mapping=PARAM_MAPPING, **model_hyperparams)
    training_name = create_kwarg_string(param_mapping=PARAM_MAPPING, **model_training, comment=comment)
    checkpoint_path = os.path.join(config["runs_dir"], architecture_name, training_name).replace("\\", "/")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch:02d}-{train_loss:.4f}-{val_loss:.4f}",
        save_top_k=-1
    )

    # Create logger.
    save_dir = os.path.join(config["logs_dir"], architecture_name).replace("\\", "/")
    logger = TensorBoardLogger(
        save_dir=save_dir, 
        name=training_name,
        default_hp_metric=False,
        version=""
    )

    # Create trainer.
    trainer = Trainer(
        max_epochs=model_training["epochs"],
        accumulate_grad_batches=model_training["accumulation_steps"],
        logger=logger,
        log_every_n_steps=5,
        callbacks=[lr_monitor, checkpoint_callback],
        precision='16',
    )

    # Create model.
    # model = FactoryModel(**model_hyperparams, **model_training)
    model = Transformer(**model_hyperparams, **model_training)

    # Train model.
    ckpt_path = os.path.join(checkpoint_path, config["checkpoint_file"]) if config["checkpoint_file"] else None 
    trainer.fit(model, dataloader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train",
        description="Train model"
    )
    parser.add_argument("--hyperparams", type=str, required=True, help="Path to hyperparams YAML file.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file.")
    parser.add_argument("--comment", type=str, default="", help="Additional comment for Tensorbord.")
    args = parser.parse_args()

    # Read YAML files.
    hyperparams = yaml_to_kwargs(args.hyperparams)
    config = yaml_to_kwargs(args.config)

    # Run training.
    train(hyperparams, config, args.comment)