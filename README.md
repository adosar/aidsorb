<h1 align="center">
  <img alt="AIdsorb logo" src="https://github.com/adosar/aidsorb/blob/master/docs/source/images/aidsorb_logo_light.svg" width=40%/>
</h1>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Roboto+Slab&weight=700&duration=3000&pause=1000&color=FFFFFF&background=000000&center=true&vCenter=true&height=40&lines=%F0%9F%9A%80+Simple%2C+easy+to+use+and+reproduce;%F0%9F%94%A5+Supports+Pytorch;%E2%9A%A1+Supports+Pytorch+Lightning;%F0%9F%8E%89+A+.yaml+is+all+you+need!" />
</p>

<h4 align="center">
  <img alt="Static Badge" src="https://img.shields.io/badge/Python%203.11%2B-black?style=for-the-badge&logo=python&logoColor=cyan&link=https%3A%2F%2Fwww.python.org%2Fdownloads%2F">
  <img alt="Static Badge" src="https://img.shields.io/badge/GPL--3.0--only-black?style=for-the-badge&logo=gnu&logoColor=cyan&link=https%3A%2F%2Fspdx.org%2Flicenses%2FGPL-3.0-only.html">
  <img alt="Static Badge" src="https://img.shields.io/badge/Linux-black?style=for-the-badge&logo=linux&logoColor=cyan">
  <img alt="Static Badge" src="https://img.shields.io/badge/Pytorch-black?style=for-the-badge&logo=pytorch&logoColor=cyan&link=https%3A%2F%2Fpytorch.org%2F">
  <img alt="Static Badge" src="https://img.shields.io/badge/Pytorch%20Lightning-black?style=for-the-badge&logo=lightning&logoColor=cyan&link=https%3A%2F%2Flightning.ai%2Fdocs%2Fpytorch%2Fstable%2F">
</h4>

**AIdsorb** is a Python package for **deep learning on molecular point clouds**.

This package aims to provide a **simple, easy-to-use and reproduce** interface for:

-   📥 **Creating molecular point clouds**
    
-   🤖 **Training DL algorithms on molecular point clouds**


## ⚙️  Installation
> [!IMPORTANT] 
> It is strongly recommended to **perform the installation inside a virtual environment**.

Assuming an activated virtual environment:
```bash
pip install aidsorb
```

## 🚀 Usage
> [!NOTE] 
> Refer to the 📚 Documentation for more information.

Here is a summary of what you can do from the command line:

1. Visualize a molecular point cloud:
	```bash
	aidsorb visualize path/to/structure
	```

2.  Create and prepare point clouds:
	```bash
	aidsorb create path/to/inp path/to/out  # Create point clouds
	aidsorb prepare path/to/out  # Split point clouds to train, val and test
	```
	where `path/to/inp` is a directory containing molecular structures.
	
3. Train and test a model:
	```bash
	aidsorb-lit fit --config=path/to/config.yaml
	aidsorb-lit test --config=path/to/config.yaml --ckpt_path=path/to/ckpt
	```
	Currently, only [PointNet](https://arxiv.org/abs/1612.00593) is supported.

## 📰 Citing AIdsorb
If you use AIdsorb in your research, please consider citing the following work:
	
	Currently, not available.

## 📑 License
AIdosrb is released under the [GNU General Public License v3.0 only](https://spdx.org/licenses/GPL-3.0-only.html).
