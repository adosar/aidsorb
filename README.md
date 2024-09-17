<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/adosar/aidsorb/master/docs/source/images/aidsorb_logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/adosar/aidsorb/master/docs/source/images/aidsorb_logo_light.svg">
    <img alt="AIdsorb logo" src="https://raw.githubusercontent.com/adosar/aidsorb/master/docs/source/images/aidsorb_logo_light.svg" width=40%/>
  </picture>
</h1>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Roboto+Slab&weight=700&duration=3000&pause=1000&color=FFFFFF&background=000000&center=true&vCenter=true&height=40&lines=%F0%9F%9A%80+Simple%2C+easy+to+use+and+reproduce;%F0%9F%94%A5+Supports+PyTorch;%E2%9A%A1+Supports+PyTorch+Lightning;%F0%9F%8E%89+A+.yaml+is+all+you+need!" />
</p>

<h4 align="center">
  <img alt="Static Badge" src="https://img.shields.io/badge/Python%203.11%2B-black?style=for-the-badge&logo=python&logoColor=cyan&link=https%3A%2F%2Fwww.python.org%2Fdownloads%2F">
  <img alt="Static Badge" src="https://img.shields.io/badge/GPL--3.0--only-black?style=for-the-badge&logo=gnu&logoColor=cyan&link=https%3A%2F%2Fspdx.org%2Flicenses%2FGPL-3.0-only.html">
  <img alt="Static Badge" src="https://img.shields.io/badge/Linux-black?style=for-the-badge&logo=linux&logoColor=cyan">
  <!--- <img alt="Static Badge" src="https://img.shields.io/badge/Pytorch-black?style=for-the-badge&logo=pytorch&logoColor=cyan&link=https%3A%2F%2Fpytorch.org%2F"> --->
  <!--- <img alt="Static Badge" src="https://img.shields.io/badge/Pytorch%20Lightning-black?style=for-the-badge&logo=lightning&logoColor=cyan&link=https%3A%2F%2Flightning.ai%2Fdocs%2Fpytorch%2Fstable%2F"> --->
  <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/adosar/aidsorb/unittest.yaml?style=for-the-badge&logo=github&logoColor=cyan&label=Tests&labelColor=black">
  <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/adosar/aidsorb/pylint.yaml?style=for-the-badge&logo=github&logoColor=cyan&label=Lint&labelColor=black">
</h4>

**AIdsorb** is a Python package for **deep learning on molecular point clouds**.

This package aims to provide a **simple, easy-to-use and reproduce** interface for:

-   📥 **Creating molecular point clouds**
  
-   🤖 **Training DL algorithms on molecular point clouds**

<p align="center">
  <img alt="IRMOF-1" src="https://raw.githubusercontent.com/adosar/aidsorb/master/docs/source/images/IRMOF-1.gif" width="25%"/>
  <img alt="Cu-BTC" src="https://raw.githubusercontent.com/adosar/aidsorb/master/docs/source/images/Cu-BTC.gif" width="25%"/>
  <img alt="UiO-66" src="https://raw.githubusercontent.com/adosar/aidsorb/master/docs/source/images/UiO-66.gif" width="25%"/>
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
> It will be available soon.

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

## 💡 Contributing

🙌 We welcome contributions from the community to help improve and expand this
project!

You can start by 🛠️ [opening an issue](https://github.com/adosar/aidsorb/issues) for:

* 🐛 Reporting bugs
* 🌟 Suggesting new features
* 📚 Improving documentation
* 🎨 Adding your example to the Gallery

We appreciate your efforts to submit well-documented 🔃 [pull
requests](https://github.com/adosar/aidsorb/pulls) and participate in
discussions.

💪 Together, we can make this project even better!


## 📑 Citing
If you use **AIdsorb** in your research, please consider citing the following work:
	
	Currently, not available.

## ⚖️ License
**AIdosrb** is released under the [GNU General Public License v3.0 only](https://spdx.org/licenses/GPL-3.0-only.html).
