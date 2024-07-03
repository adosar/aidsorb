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
#         * Training was performed with AIdsorb :doc:`../cli` or :ref:`AIdsorb +
#           PyTorch Lightning <aidsorb_with_pytorch_and_lightning>`.

import yaml
import torch
import lightning as L
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from aidsorb.datamodules import PCDDataModule
from aidsorb.litmodels import PointLit
from aidsorb.visualize import draw_pcd

# %%
# The following function let us recreate:
# 
# * Trainer
# * LightningModule (litmodel)
# * Datamodule
# 
# with the same settings as in the ``.yaml`` configuration file. For more
# information 👉 `here
# <https://github.com/Lightning-AI/pytorch-lightning/discussions/10363#discussioncomment-2326235>`_.

def load_from_config(filename):
    r"""
    Load configuration, trainer, model and datamodule from a ``.yaml`` file.

    .. note::
        You are responsible for restoring the model's state (the weights of the model).

    Parameters
    ----------
    filename: str
        Absolute or relative path to the ``.yaml`` configuration file.
    """
    with open(filename, 'r') as f:
        config_dict = yaml.safe_load(f)

    config_dict['trainer']['logger'] = False
    del config_dict['seed_everything'], config_dict['ckpt_path']

    parser = LightningArgumentParser()
    parser.add_class_arguments(PointLit, 'model', fail_untyped=False)
    parser.add_class_arguments(PCDDataModule, 'data', fail_untyped=False)
    parser.add_class_arguments(L.Trainer, 'trainer', fail_untyped=False)

    config = parser.parse_object(config_dict)
    objects = parser.instantiate_classes(config)

    return config, objects.trainer, objects.model, objects.data

# %%

config, trainer, litmodel, dm = load_from_config('path/to/logs/config.yaml')

# %%  The remaining part is to restore the model's state, i.e. load back the trained weights.

# %%
# Restoring model's state 
# -----------------------

ckpt = torch.load('path/to/checkpoints/checkpoint.ckpt')
model_weights = {k: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}

# %%

# Due to lazy initialization we need to pass a dummy input with correct shape.
in_channels = 5  # For xyz + Z + 1 additional feature.
x = torch.randn(32, in_channels, 100)
litmodel(x);

# %%

# Load back the weights.
litmodel.load_state_dict(model_weights)

# %%

# Set the model in inference mode.
litmodel.eval()
litmodel.training


# %%
# Measure performance and make predictions
# ----------------------------------------

# Measure performance on test set.
trainer.test(litmodel, dm)

# %%

# Predict on the test set.
y_pred = torch.cat(trainer.predict(litmodel, dm.test_dataloader()))

# Predict on the train set.
y_pred = torch.cat(trainer.predict(litmodel, dm.train_dataloader()))
