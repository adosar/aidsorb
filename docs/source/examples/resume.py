r"""
Coming back after model training
================================
"""

# %%
# After training a model, you might want to test its performance, make
# predictions or do whatever you want with it.
#
# .. note::
#     This example assumes:
#         * Training was performed with :doc:`AIdsorb CLI <../cli>` or :ref:`AIdsorb +
#           PyTorch Lightning <aidsorb_with_pytorch_and_lightning>`.
#         * `PyTorch Lightning checkpoints
#           <https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#lightningmodule-from-checkpoint>`_
#           are enabled during training.

import lightning as L
import torch

from aidsorb.datamodules import PCDDataModule
from aidsorb.litmodules import PCDLit

# Restore lightning modules from checkpoint.
ckpt_path = 'path/to/checkpoint.ckpt'
litmodel = PCDLit.load_from_checkpoint(ckpt_path)
dm = PCDDataModule.load_from_checkpoint(ckpt_path)

# Set the model for inference (disable grads & enable eval mode).
litmodel.freeze()
print(f'Model in evaluation mode: {not litmodel.training}')

# Your code goes here.
...


# %%
# Measure performance
# -------------------

# Instantiate a trainer object.
trainer = L.Trainer(...)

# Measure performance on test set.
trainer.test(litmodel, datamodule=dm)

# %%
# Make predictions
# ----------------

# Setup the datamodule.
dm.setup()

# Predict on the test set.
y_pred = torch.cat(trainer.predict(litmodel, dm.test_dataloader()))

# Predict on the train set.
y_pred = torch.cat(trainer.predict(litmodel, dm.train_dataloader()))
