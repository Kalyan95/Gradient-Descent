# Gradient Descent Classifier

This project demonstrates logistic regression from scratch using gradient descent.
A two-class dataset is generated with `sklearn.datasets.make_moons` and a simple
training loop optimises the weights.

## Requirements

- Python 3.7+
- `numpy`
- `scikit-learn`
- `matplotlib`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

The main code lives in `GradientDescent.ipynb`. Launch Jupyter and run all cells:

```bash
jupyter notebook GradientDescent.ipynb
```

The notebook will:

1. Create a dataset with Gaussian noise
2. Train a binary classifier for a set number of epochs
3. Output accuracy and a loss curve

You can adapt the parameters `epochs` and `alpha` inside the notebook to
experiment with the optimisation process.