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