r"""
Fine-tune a pretrained model
============================
"""

# %%
# This example demonstrates how to fine-tune a pretrained model.
#
# Fine-tuning differs slightly from training a model from scratch. It requires
# additional configuration, including loading pretrained weights, selecting the
# layers to freeze, and setting layer-specific learning rates (typically lower
# than in training from scratch).
#
# |aidsorb| provides full control over the fine-tuning process while reducing
# boilerplate code thanks to its integration with |lightning| (e.g.,
# training loops, checkpointing, and GPU handling).
#
# In this example, we fine-tune :class:`.IntelliPore`, a pretrained model included
# in the package. IntelliPore takes energy images as input and can be adapted to
# predict adsorption properties through fine-tuning.
#
# .. tip::
#    The same workflow can be applied to any custom pretrained model, as long as
#    it is implemented as a :class:`torch.nn.Module`.
#
# Dataset preparation
# -------------------
#
# The first step is to prepare the dataset and split it into training, validation,
# and test sets. This can be easily done using the :doc:`AIdsorb CLI <../cli>`.
#
# It is important to **ensure that the input data is generated using the same parameters
# expected by the pretrained model**.
#
# .. code-block:: console
#
#    $ aidsorb create voxels path/to/CIFs path/to/voxels_data --grid_size=32 --cubic_box=30
#    $ aidsorb prepare path/to/voxels_data/ --split_ratio='[0.8, 0.1, 0.1]' --seed=42

# %%
# Model fine-tuning
# -----------------

import torch
from torchvision.transforms.v2 import Compose, RandomChoice
from torchmetrics import R2Score, MeanAbsoluteError, MetricCollection
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from aidsorb.datamodules import DataModule
from aidsorb.litmodules import LitModule
from aidsorb.modules.voxels import IntelliPore
from aidsorb.transforms.voxels import (
    AddChannelDim, ClipScaleVoxels,
    RandomRotate90, RandomReflect, RandomFlip,
)


# Custom optimizer with layer-wise learning rates.
def custom_optimizer(self):
    return torch.optim.Adam([
        {'params': self.model.backbone.parameters(), 'lr': 1e-3},
        {'params': self.model.head.parameters(), 'lr': 1e-2},
    ])


# For reproducibility.
seed_everything(42, workers=True)

# Load pretrained model and freeze early backbone layers.
model = IntelliPore(n_outputs=1, pretrained=True)
model.backbone[:6].requires_grad_(False)
model.backbone[:6].eval()

# Preprocessing and augmentation transformations.
# IMPORTANT: use same preprocessing as the pretrained model.
eval_transform = Compose([AddChannelDim(), ClipScaleVoxels()])
train_transform = Compose([
    AddChannelDim(),
    ClipScaleVoxels(),
    RandomChoice([
        torch.nn.Identity(),
        RandomRotate90(),
        RandomFlip(),
        RandomReflect(),
    ]),
])

# Overwrite the default optimizer.
LitModule.configure_optimizers = custom_optimizer

# Define the loss and evaulation metrics.
criterion = torch.nn.MSELoss()
metric = MetricCollection(R2Score(), MeanAbsoluteError())

# Create the litmodule.
litmodel = LitModule(model, criterion, metric=metric)

# Create the datamodule.
datamodule = DataModule(
    path_to_X='path/to/voxels_data/',
    path_to_Y='path/to/labels.csv',
    index_col='id',
    labels=['adsorption_property'],
    train_batch_size=32,
    eval_batch_size=256,
    train_transform_x=train_transform,
    eval_transform_x=eval_transform,
    shuffle=True,
    drop_last=True,
    config_dataloaders=dict(num_workers=8),
)
datamodule.setup()

# Enable model checkpointing to avoid overfitting.
checkpoint_callback = ModelCheckpoint(
    monitor='val_R2Score',
    mode='max',
    filename='best',
    save_top_k=1,
)

# Create the trainer.
trainer = Trainer(
    max_epochs=100,
    accelerator='gpu',
    callbacks=checkpoint_callback,
)

# Initialize output bias with training mean (optional but recommended).
y_mean = datamodule.train_dataset.Y.mean().item()
torch.nn.init.constant_(model.head.bias, y_mean)

# Train and test the fine-tuned model.
trainer.fit(litmodel, datamodule=datamodule)
trainer.test(litmodel, datamodule=datamodule, ckpt_path='best')
