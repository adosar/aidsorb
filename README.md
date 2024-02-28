AIdsorb is a Python package for processing molecular point clouds.

Currently, only the `PointNet`  [^1] architecture is implemented and can be
found under `src/aidsorb/models.py`. In this implementation, the input `T-Net`
transformation from the original paper is not applied, *since is it is not
guaranteed to be a rigid one*.

## ⚙️  Installation
> Currently not available in PyPI.

**It is strongly recommended to perform the installation inside a virtual environment**.
```bash
git clone https://github.com/adosar/aidsorb && cd aidsorb
(venvir) pip install -e .
```

## 📖 Usage
You can create molecular point clouds form a directory containing molecular
files and save them in a `.npz` file.

```python
from aidsorb.utils import pcd_from_dir
pcd_from_dir('/path/to/dir', file='point_clouds.npz')
```

## 📇 TODO
* Add atoms featurization
* Add more architectures

## 📰 Citing AIdsorb
> Currently N/A.

## 📑 License
MOXελ is released under the [GNU General Public License v3.0 only](https://spdx.org/licenses/GPL-3.0-only.html).

[^1]: R. Q. Charles, H. Su, M. Kaichun and L. J. Guibas, "PointNet: Deep
Learning on Point Sets for 3D Classification and Segmentation," 2017 IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI,USA,
2017, pp. 77-85, doi: 10.1109/CVPR.2017.16.
