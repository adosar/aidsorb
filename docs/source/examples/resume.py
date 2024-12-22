r"""
Coming back after model training
================================
"""

# %%
# After training a model, you might want to test its performance, make
# predictions or do whatever you want with it.
#
# .. note::
#     This example assummes:
#         * `PyTorch Lightning checkpoints
#           <https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#lightningmodule-from-checkpoint>`_
#           are enabled during training.
#         * Training was performed with :doc:`AIdsorb CLI <../cli>` or :ref:`AIdsorb +
#           PyTorch Lightning <aidsorb_with_pytorch_and_lightning>`.

import yaml
import torch
import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
from aidsorb.datamodules import PCDDataModule
from aidsorb.litmodels import PCDLit

# %%
# The following snipper let us instantiate:
# 
# * :class:`~lightning.pytorch.trainer.trainer.Trainer`
# * :class:`~lightning.pytorch.core.LightningModule`
# * :class:`~lightning.pytorch.core.LightningDataModule`
# 
# with the same settings as in the ``.yaml`` configuration file. For more
# information 👉 `here
# <https://github.com/Lightning-AI/pytorch-lightning/discussions/10363#discussioncomment-2326235>`_.

# %%
# .. note::
#     You are responsible for restoring the model's state (the weights of the model).

filename = 'path/to/config.yaml'
with open(filename, 'r') as f:
    config_dict = yaml.safe_load(f)

parser = LightningArgumentParser()
parser.add_lightning_class_args(PCDLit, 'model')
parser.add_lightning_class_args(PCDDataModule, 'data')
parser.add_class_arguments(L.Trainer, 'trainer')

# Any other key present in the config file must also be added.
# parser.add_argument(--<keyname>, ...)
# For more information 👉 https://jsonargparse.readthedocs.io/en/stable/#parsers
parser.add_argument('--seed_everything')
parser.add_argument('--ckpt_path')
parser.add_argument('--optimizer')
parser.add_argument('--lr_scheduler')

config = parser.parse_object(config_dict)
objects = parser.instantiate_classes(config)

# %%

trainer, litmodel, dm = objects.trainer, objects.model, objects.data

# %%  The remaining part is to restore the model's state, i.e. load back the trained weights.

# %%
# Restoring model's state 
# -----------------------

# Load the the checkpoint.
ckpt = torch.load('path/to/checkpoints/checkpoint.ckpt')

# %%

# Load back the weights.
litmodel.load_state_dict(ckpt['state_dict'])

# %%

# Set the model for inference (disable grads & enable eval mode).
litmodel.freeze()
print(f'Model in evaluation mode: {not litmodel.training}')

# Your code goes here.
...


# %%
# Measure performance
# -------------------

# Measure performance on test set.
trainer.test(litmodel, dm)

# %%
# Make predictions
# ----------------

# Setup the datamodule.
dm.setup()

# Predict on the test set.
y_pred = torch.cat(trainer.predict(litmodel, dm.test_dataloader()))

# Predict on the train set.
y_pred = torch.cat(trainer.predict(litmodel, dm.train_dataloader()))
