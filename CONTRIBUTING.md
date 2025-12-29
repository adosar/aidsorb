# 🙌 Contributing

 We welcome contributions from the community to help improve and expand this
project!

You can start by 🛠️ [opening an issue](https://github.com/adosar/aidsorb/issues) for:

* 🐛 Reporting bugs
* 🌟 Suggesting new features
* 📚 Improving documentation
* 🎨 Adding your example to the [Gallery](https://aidsorb.readthedocs.io/en/stable/auto_examples/index.html)

## Before You Start

1. Check for existing issues or pull requests to avoid duplicates.
2. Small fixes (like doc typos) can be submitted directly via a pull request.
3. Larger changes or new features should have an associated issue first.

## Pull Requests

To contribute your changes follow these steps:

1. **[Fork the original repository][forking]**

2. **Clone the forked repository**

    ```bash
    git clone https://github.com/<github-user>/aidsorb.git
    cd aidsorb
    ```

3. **Set up a development environment**

    Run the following command inside a virtual environment:

    ```bash
    pip install -e .
    ```

4. **Create a new branch**

    ```bash
    git switch -c <branch-name>
    ```

    Although optional, it is good practice to use a descriptive name for the new branch:

    * General changes: `<type>/<short-description>`
    * Linked to an issue: `<type>/<issue-id>_<short-description>`

    Examples of branch names:

    ```bash
    git switch -c docs/add-classification-example
    git switch -c feat/add-architecture-foo
    git switch -c fix/42_fix-forward-bar
    ```

5. **Make your changes**

    * Add or edit code, tests, or documentation as needed.
    * Ensure your changes follow the [style guide](#style-guide).

6. **Commit your changes**

    Before you commit:

    * If you are modifying code, make sure all tests pass. See here 👉 how to
      [run tests](#tests).
    * If you are modifying documentation, make sure the docs build successfully.
      See here 👉 how to [build docs](#documentation).
    * If you are modifying both, make sure tests pass and docs build
      successfully.

    Use [Conventional Commits format][conventional-commits]:

    ```bash
    git add <modifications>
    git commit -m "<conventional-commit-message>"
    ```

7. **Push your branch to your fork**

    ```bash
    git push origin <branch-name>
    ```
    
    * Go to your fork on GitHub and click the “Compare & pull request” button.
    * Provide a clear description of what your PR does and reference related issues (if any), e.g. "Fixes #42".

## Tests

It is important to test code. Whenever you **add a new feature or modify
existing code**, you should ensure that it is covered by tests.

For example, if you are adding a new function `foo` in the module `bar.py`, you should:

* Add unit test(s) using the `unittest` package in `tests/test_bar.py` **and/or**
* Add doctest(s) inside the module.

To run unit tests, the following commands should be executed from the project's root directory:

```bash
python -m unittest tests.<test_name>  # run test for an individual module
python -m unittest  # run all tests at once
```

Useful links:  
* [`unittest` documentation][unittest]
* [`doctest` documentation][doctest]

## Documentation

It is important to keep the documentation up to date. Whenever you **add a new feature or modify existing code**,
make sure the changes are reflected in the documentation.

To build the docs:

```bash
pip install '.[docs]'
cd docs
make html  # or make clean && make html to buld from scratch
```

To view the docs:

```bash
<browser> build/html/index.html
```

Useful links:
* [Sphinx documentation][sphinx]
* [reStructuredText tutorial][rst]

## Style Guide

* Follow [NumPy docstring conventions][numpydoc] for documentation.
* Use [Conventional Commits][conventional-commits] for your commit messages.
* Adhere to [PEP 8][pep8] for code style.

[unittest]: https://docs.python.org/3/library/unittest.html
[doctest]: https://docs.python.org/3/library/doctest.html
[forking]: https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project
[conventional-commits]: https://www.conventionalcommits.org/en/v1.0.0/
[numpydoc]: https://numpydoc.readthedocs.io/en/latest/format.html
[pep8]: https://peps.python.org/pep-0008/
[sphinx]: https://www.sphinx-doc.org/en/master/index.html
[rst]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#rst-primer
