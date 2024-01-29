# Contributing to AudioSeal

We want to make contributing to AudioSeal as easy as possible. Please make sure
to read this guideline carefully.

## Setting up Development Environment

AudioSeal is a lightweight Python library that only relies on PyTorch, Numpy and OmegaConf (for
model card loading). Currenet minimal Pytorch requirement is 13.0, and it is advisable to
keep the constraints on PyTorch as lenient as possible. Please keep both the text file
`requirements.txt` and the Poetry file `pyproject.toml` up-to-date should you change the
third-party library requirements.

```sh
git clone https://github.com/facebookresearch/audioseal.git
```

And, install the package in editable mode with development tools before contributing:

```sh
cd audioseal
pip install -e ".[dev]"
```

Alternatively, you can also install the package and its development tools separately

```sh
cd audioseal
pip install -e .
pip install -r requirements-dev.txt
```

It is advisable to keep your commits linted and syntax-correct. In AudioSeal we provide a few
[pre-commit] hooks to support that. Simply install pre-commit:

```sh
pre-commit install .
```

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License

By contributing to `SONAR`, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
