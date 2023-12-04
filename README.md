### Benchmarking and evaluation tools

This repo contains software that generated our benchmarking results. For more info on this project, see the [benchmarking results repo](https://github.com/ekernf01/perturbation_benchmarking).

### Installation

Just this package:

`pip install 'perturbation_benchmarking_package @ git+https://github.com/ekernf01/perturbation_benchmarking_package.git'`

With our other related packages:

```bash
for p in load_networks load_perturbations ggrn_backend2 ggrn perturbation_benchmarking_package geneformer_embeddings
do
    pip install ${p} @ http://github.com/ekernf01/${p}
done
```

### Adding your own metrics

To add your own evaluation metrics, you will need to make a fork of this repo, edit `evaluator.py`, and install it prior to running your experiments. Near the top of `evaluator.py` is a dict `METRICS` containing our evaluation functions. You can add any function following the same format. Results will be included in a column named after the key you add to the dictionary.  