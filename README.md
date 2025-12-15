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
	
  ![Static Badge](https://img.shields.io/badge/PYTHON%203.11+-black?style=for-the-badge&logo=python&logoColor=cyan)
  ![Static Badge](https://img.shields.io/badge/GPL--3.0--ONLY-black?style=for-the-badge&logo=gnu&logoColor=cyan)
  ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/adosar/aidsorb/unittest.yaml?style=for-the-badge&logo=github&logoColor=cyan&label=TESTS&labelColor=black)
  ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/adosar/aidsorb/pylint.yaml?style=for-the-badge&logo=github&logoColor=cyan&label=LINT&labelColor=black)
  ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/adosar/aidsorb/build-install.yaml?style=for-the-badge&logo=github&logoColor=cyan&label=WHEEL&labelColor=black)
  [![coverage](https://img.shields.io/codecov/c/gh/adosar/aidsorb?style=for-the-badge&logo=codecov&logoColor=cyan&label=CODECOV&labelColor=black&color=purple)](https://app.codecov.io/gh/adosar/aidsorb)
  [![Docs](https://img.shields.io/badge/foo-stable-black?style=for-the-badge&logo=readthedocs&logoColor=cyan&label=ReadTheDocs&labelColor=black&color=purple)](https://aidsorb.readthedocs.io/en/stable/)
  [![PyPI](https://img.shields.io/pypi/v/aidsorb?style=for-the-badge&logo=pypi&logoColor=cyan&labelColor=black&color=purple)](https://pypi.org/project/aidsorb/)
  [![App](https://img.shields.io/badge/online%20app-purple?style=for-the-badge&logo=streamlit&logoSize=auto&logoColor=cyan&label=streamlit&labelColor=black)](https://aidsorb-online.streamlit.app)

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
> Refer to the 📚 [Documentation](https://aidsorb.readthedocs.io/en/stable/) for more information.

Here is a summary of what you can do from the command line:

1. Visualize a point cloud:
	```bash
	aidsorb visualize path/to/structure_or_pcd  # Structure (.xyz, .cif, etc) or .npy
	```

2.  Create and prepare point clouds:
	```bash
	aidsorb create path/to/structures path/to/pcd_data  # Create and store point clouds
	aidsorb prepare path/to/pcd_data  # Split point clouds to train, valdation and test
	```
	
3. Train and test a model:
	```bash
	aidsorb-lit fit --config=path/to/config.yaml
	aidsorb-lit test --config=path/to/config.yaml --ckpt_path=path/to/ckpt
	```
 
## 💡 Questions and Contributing

### Questions
If you have any questions about how to use **AIdsorb**, we encourage you to post them in the 💬 [Discussions](https://github.com/adosar/aidsorb/discussions)
section of the repository.

> [!NOTE]
> Please make sure to **read the documentation carefully first** before asking your question.

### Contributing
We welcome contributions from the community! Please read our 🙌 [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs or opening issues.

## 📑 Citing
* **To cite the software**, please refer to the [citation file](./CITATION.cff) or click the citation button.
* **To cite the paper**, please use the following BibTeX entry:
<details>
<summary>Show BibTex entry</summary>
	
```bibtex
@article{Sarikas2024,
  title = {Gas adsorption meets geometric deep learning: points, set and match},
  volume = {14},
  ISSN = {2045-2322},
  url = {http://dx.doi.org/10.1038/s41598-024-76319-8},
  DOI = {10.1038/s41598-024-76319-8},
  number = {1},
  journal = {Scientific Reports},
  publisher = {Springer Science and Business Media LLC},
  author = {Sarikas,  Antonios P. and Gkagkas,  Konstantinos and Froudakis,  George E.},
  year = {2024},
  month = nov
}
```
</details>

## ⚖️ License
**AIdosrb** is released under the [GNU General Public License v3.0 only](https://spdx.org/licenses/GPL-3.0-only.html).
