### Benchmarking and evaluation tools

This repo contains software that generated our benchmarking results. For full documentation, see the [benchmarking results repo](https://github.com/ekernf01/perturbation_benchmarking).

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
