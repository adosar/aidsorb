<h1 align="center">
  <img alt="Logo" src="https://raw.githubusercontent.com/adosar/trial/master/docs/source/images/aidsorb_logo_light.svg"/>
</h1>

<h4 align="center">

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white&labelColor=black)
![License GPL-3.0-only](https://img.shields.io/badge/license-GPL--3.0--only-gold?style=for-the-badge&logo=license&logoColor=cyan&labelColor=black)
![Linux](https://img.shields.io/badge/linux-black?style=for-the-badge&logo=linux&logoColor=cyan)

</h4>


**AIdsorb** is a Python package for **deep learning on molecular point clouds**.

**AIdsorb** aims to provide a **fast, easy-to-use and reproduce** interface for:

-   📥 **Creating molecular point clouds**
    
-   🤖 **Training DL algorithms on molecular point clouds**


<p align="center">
  <img alt="Point cloud" src="https://raw.githubusercontent.com/adosar/aidsorb/master/docs/source/images/pcd.gif" width="25%"/>
</p>

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
