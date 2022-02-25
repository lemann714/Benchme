# Automatic benchmarking of classifiers

Package provides simple interactive benchmarking of classifiers for particular data.

The proccess of selecting suitable model can be divided on two steps.
First off all optimizers are cross-validated with default parameters. Secondly, user is prompted to give desired metric, which is tested on all optimizers with varied parameters combination. The best model as well the best parameters are printed.

Clone the repository
```
git clone https://github.com/lemann714/Benchme.git
```
Then ```cd``` into the root directory of the project.

You now can run benchmarking by typing in your shell ```./bench.py FILENAME```, where FILENAME is a source file of csv format.

positional arguments:
  source                CSV-data

optional arguments:
  -h, --help            show this help message and exit
  --target TARGET, -t TARGET
                        Index of target column
  --sep SEP, -s SEP     Columns separator symbol
