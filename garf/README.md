# Self-supervised and Interpretable Data Cleaning with Sequence Generative Adversarial Networks

Original research code is at https://github.com/PJinfeng/Garf-master. We slightly adapted this code base to fit in our experimental setup.


## Build Docker Container

1. build the `conformal-data-cleaning` package as wheel: `poetry build`
   - make sure to only use the following dependencies in `pyproject.toml`
   - `python = "^3.7,<3.10"`
   - `scikit-learn = "<1.3.0"`
2. move the `dist` directory to `garf/dist`
3. `docker build -t conformal-data-cleaning:garf .`

