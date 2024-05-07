### Benchmarking and evaluation tools

This repo contains software that generated our benchmarking results. For full documentation, see the [benchmarking results repo](https://github.com/ekernf01/perturbation_benchmarking).

### Usage

This code is currently "trapped" in the very specific file/folder layout of our benchmarking repo mentioned above. Soon, we hope to allow a more flexible interface, similar to the following. 

`pereggrn --config metadata.json --output experiments`

The contents of `metadata.json` will describe what networks to test, how many genes to include, and tons of other information about the experiment. It is documented [here](https://github.com/ekernf01/perturbation_benchmarking/blob/main/docs/reference.md#metadata-specifying-an-experiment) although this may get rearranged soon to make it easier to reuse. 

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
