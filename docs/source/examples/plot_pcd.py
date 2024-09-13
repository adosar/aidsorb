r"""
Create and visualize a molecular point cloud
============================================
"""
from plotly.io import show
from aidsorb.utils import pcd_from_file
from aidsorb.visualize import draw_pcd

# sphinx_gallery_thumbnail_number = 2

# %%
# Create the point cloud
# ----------------------

# %%

# We add electronegativity as additional feature to atomic number.
name, pcd = pcd_from_file('IRMOF-1.xyz', features=['en_pauling'])

print(f'The name of the point cloud is: {name}')
print(f'The point cloud has shape: {pcd.shape}')

# %%
# Visualize it
# ------------

fig = draw_pcd(pcd, scheme='jmol')
fig.update_layout(margin=dict(b=0, t=0, l=0, r=0))  # Optional.
show(fig)

# %%
# .. tip::
#    
#     You can also use the CLI:
#
#     .. code-block:: console
#
#         $ aidsorb visualize path/to/structure

# %%

# Color it by electronegativity.
fig = draw_pcd(pcd, feature_to_color=(4, 'Electronegativity'), colorscale='viridis')
fig.update_layout(margin=dict(b=0, t=0, l=0, r=0))  # Optional.
show(fig)
