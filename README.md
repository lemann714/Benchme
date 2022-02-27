# Automatic benchmarking of classifiers

## Info
Package provides simple interactive benchmarking of classifiers for particular data.

The proccess of selecting suitable model can be divided on two steps.
First off all optimizers are cross-validated with default parameters. Secondly, user is prompted to give desired metric, which is tested on all optimizers with varied parameters combination. The best model as well the best parameters are printed.

## Usage example
Clone the repository
```
git clone https://github.com/lemann714/Benchme.git
```
Then ```cd``` into the root directory of the project.

You now can run benchmarking by typing in your shell ```./bench.py FILENAME```, where FILENAME is a positional argument, indicating source file of csv format.

Among optional arguments are:

- -s, --sep: set columns separator symbol
- -t, --target: set index of target column
- -h, --help: show help message and exit

Running ```./bench.py FILENAME``` script will show up table contaning comparison of classifiers performance given default parameters. After table print user is prompted whether to input metric to order classifiers by column or to proceed to best model and parameters search process with following message: `Enter column name to order classifiers. Press [p] to optimal model search, [Ctrl+c] to exit`. 
After pressing `[p]` user is prompted to input desired metric with message: `Choose metric from ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']. Press [Ctrl+c] to exit`.

## Usefull information
In order to find best parameters RandomizedSearchCV is used. Bounds of hyperparameters can be set in settings.py module.
