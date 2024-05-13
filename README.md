### PEREGGRN: PErturbation Response Evaluation via a Grammar of Gene Regulatory Networks

This repo contains software that generated our [benchmarking results](https://github.com/ekernf01/perturbation_benchmarking). . 

### Usage

Consult `pereggrn -h`, or the [how-to recipes](https://github.com/ekernf01/perturbation_benchmarking/blob/main/docs/how_to.md), or the [full reference](https://github.com/ekernf01/perturbation_benchmarking/blob/main/docs/reference.md).

In brief: pereggrn will look for a `metadata.json` that describes what data to load, how to do the train-test split(s), what network(s) to test, how many genes to include, hyperparameter(s) to use, and tons of other information about the experiment. It will load in data and networks, split the data, train models, make predictions, and score the predictions. The main outputs are full expression forecasts, evaluation results, and a small table describing the different conditions included in the experiment.

### Installation

Just this package:

`pip install 'pereggrn @ git+https://github.com/ekernf01/pereggrn.git'`

With our other related packages:

```bash
for p in pereggn_networks pereggrn_perturbations pereggrn ggrn ggrn_backend2 geneformer_embeddings
do
    pip install ${p} @ http://github.com/ekernf01/${p}
done
```
