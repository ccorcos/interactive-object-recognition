# Dice

This package contains functions for modeling the manipulation and observation of dice. This package is used for research in interactive object recognition.

## Getting Started

The `Die` model depend on the `se3` package to model its orientation.

    pip install se3

This package also uses `sparkprob` for visualizing the probability distribution across actions:

    pip install sparkprob

This package is not on PyPI so if you want to use it, you should install it in development mode so you can change the code:

    python setup.py develop

To unlink this package when you are done with it

    python setup.py develop --uninstall
