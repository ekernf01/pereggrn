### PEREGGRN: PErturbation Response Evaluation via a Grammar of Gene Regulatory Networks

The `pereggrn` software was written for an [expression forecasting benchmark project](https://github.com/ekernf01/perturbation_benchmarking). It compares predicted and observed expression under novel genetic perturbations. In this tutorial, you will reproduce a portion of a figure panel from scratch: you'll download data, install software, configure an experiment, run the experiment, and plot the results. 

#### Working directory

You need a Mac or Linux computer for this. Set up a folder `expression_forecasting_benchmarks`. This will be the working directory for all commands unless otherwise noted. 

```sh
mkdir expression_forecasting_benchmarks
cd expression_forecasting_benchmarks
```

#### Download data

Download our data from [zenodo](https://doi.org/10.5281/zenodo.8071808). You need `accessory_data`, `perturbation_data_minimal`, and `network_collection_minimal`. 

```sh
wget https://zenodo.org/record/13345104/files/accessory_data.zip  && unzip accessory_data.zip 
wget https://zenodo.org/records/13785607/files/perturbation_data_minimal.zip && unzip perturbation_data_minimal.zip 
wget https://zenodo.org/records/13785607/files/network_collection_minimal.zip && unzip network_collection_minimal.zip
```

#### Install software

You need `conda` or `miniconda` for this. Download our code and install it in a conda environment called `ggrn_minimal`. 

```sh
conda create -n ggrn_minimal
conda activate ggrn_minimal
conda install -y pip
pip install vl-convert-python
pip install ray[tune]
for p in ggrn pereggrn pereggrn_networks pereggrn_perturbations 
do
    pip install git+https://github.com/ekernf01/${p}
done
```

#### Configure an experiment

By convention, we store configuration files and results in `perturbation_benchmarking/experiments/<your_experiment_name>`. Create an empty folder at this path -- or download our other results. Create a blank `metadata.json` file.

```sh
git clone https://github.com/ekernf01/perturbation_benchmarking #optional
mkdir -p perturbation_benchmarking/experiments/tutorial
touch perturbation_benchmarking/experiments/tutorial/metadata.json
open perturbation_benchmarking/experiments/tutorial/metadata.json
```

In `metadata.json`, give your experiment a unique id, a nickname, and a longer description (readme). If you have a numbered list of guiding questions, you can refer to that too. Specify a dataset name (we will use `nakatake`), the number of genes to select (we will use 1000), and some methods to compare. For the causal structure, we will use a fully connected network ("dense"). The contents should look like this. 

```json
{
    "unique_id": "tutorial",
    "nickname": "Mar Sara colony",
    "readme": "Demo experiment for the tutorial",
    "is_active": true,
    "factor_varied": "regression_method",
    "question": "1.0",
    "regression_method":[
        "mean",
        "median",
        "RidgeCV"
    ],
    "num_genes": 1000,
    "perturbation_dataset": "nakatake",
    "data_split_seed": 42
}
```

#### Run the experiment

Run it. 

```bash
conda activate ggrn_minimal
pereggrn --experiment_name tutorial --input perturbation_benchmarking/experiments --output perturbation_benchmarking/experiments --amount_to_do missing_models --networks network_collection_minimal/networks --data perturbation_data_minimal/perturbations
```

In `perturbation_benchmarking/experiments/tutorial`, it should create `outputs`. 

```
├── conditions.csv
├── evaluationPerPert.parquet
├── evaluationPerTarget.parquet
├── genes_modeled.csv
└── predictions
    ├── 0.h5ad
    ├── 1.h5ad
    └── 2.h5ad
```

#### Plot the results

You can do this however you prefer. We use ggplot2 and a few other common R packages.

```R
library("ggplot2")
library("arrow")
library("magrittr")
library("dplyr")
library("tidyr")
filepath = paste0("~/Downloads/tutorial/outputs/evaluationPerPert.parquet")
X <- arrow::read_parquet(filepath, as_data_frame = T, mmap = T)
metrics = c(   
  "pearson_top_100", 
  "overlap_top_100",                 
  "mse_top_100", 
  "mse", 
  "mae", 
  "spearman", 
  "proportion_correct_direction", 
  "cell_type_correct"
)
metrics_where_bigger_is_better = c(
  "spearman", "proportion_correct_direction",                 
  "pearson_top_100", 
  "overlap_top_100",                 
  "proportion correct direction",
  "cell_type_correct"
) 

get_percent_improvement_over_mean = function(value, method, baseline_method = "mean"){
  baseline_performance = value[method==baseline_method]
  100*(value - baseline_performance) / abs(baseline_performance)
}
long_format = X %>% 
  group_by(regression_method) %>%
  summarise(across(metrics, mean)) %>%
  tidyr::pivot_longer(cols = all_of(metrics), names_to = "metric")
long_format %<>%
  mutate(value = value*ifelse(metric %in% metrics_where_bigger_is_better, 1, -1)) %>%
  mutate(percent_improvement_over_mean = get_percent_improvement_over_mean(value, regression_method))
ggplot(long_format) +
  geom_tile(aes(x = regression_method, y = metric, fill = percent_improvement_over_mean)) + 
  labs(x = "", y = "", fill = "Percent improvement over mean") +
  scale_fill_gradient2(mid = "grey90", low = "blue", high = "yellow") 
```

The result should look like the top left portion of figure S3.

### Next steps

For recipes on how to run specific types of experiments, add your own data, test a new method, and more, we have a [how-to guide](https://github.com/ekernf01/pereggrn/blob/main/docs/how_to.md). For a comprehensive manual, see the [reference](https://github.com/ekernf01/pereggrn/blob/main/docs/reference.md). To collaborate or get help, contact us via email or by filing a Github issue. We'd love to make this resource useful to anyone studying or predicting transcription.