## March-May 2024 updates to the benchmarking framework for timeseries prediction

This document is for internal use as we adapt the whole benchmarking framework to use time-series data. This document assumes close familiarity with the state of the code circa March 2024. If you are reading this in the future and you need additional context or explanation, contact me (Eric). 

Here's what we want.

- We want to be able to make predictions for any combo of `cell_type`, `timepoint`, `perturbation`, `expression_level_after_perturbation`, `prediction_timescale`. 
- For existing experiments, default behavior is to make one prediction per test sample, with different `expression_level_after_perturbation`, `perturbation` but same `cell_type`, `timepoint`, `prediction_timescale`. We want to keep this default behavior.
- For `timeseries` data splits, we want new default behavior. This will be to make predictions for all combos of `cell_type`, `timepoint`, `perturbation`, `prediction_timescale` but only the average `expression_level_after_perturbation`. This allows for the necessary complexity of timeseries training data while minimizing computational burden. 
- The interface between GGRN and benchmarking code will have to change -- it cannot currently carry all this info. 
- The evaluation code will have to change, since the shape of the output will change and there are limits on how literally we can interpret the output. More detail below. 
- For methods where `does_simulation_progress` is True (explanation below), one can make non-trivial predictions about control samples. Each backend should run forward simulation of controls whenever `expression_level_after_perturbation` is missing, or whenever `perturbation` is not a gene name in `.var`, or when both those conditions hold. **TODO: audit timeseries backends for behavior on controls, and include controls in the default timeseries data split.**

### New interfaces for making predictions 

#### Pereggrn call to ggrn

- in GRN.predict(), the `perturbations` arg will be replaced by a new arg called `predictions_metadata`. It will be a pandas dataframe with columns `cell_type`, `timepoint`, `perturbation`, `perturbation_type`, `is_control`, and `expression_level_after_perturbation`. 
    - It will default to `predictions.obs` or `starting_expression.obs` if those are provided.
    - It will be provided to Dockerized methods as a csv. 
    - The meaning is "predict expression in `cell_type` at `time_point` if `perturbation` were set to `expression_level_after_perturbation`". However, methods may ignore some columns. For all aim 2 benchmarks, columns `cell_type`, `timepoint` will be placeholders and will be ignored. For GEARS, `expression_level_after_perturbation` is ignored. 
    - The old `perturbations` arg will be completely removed, breaking backwards compatibility.

#### GGRN call to backends

- `predictions_metadata` will be passed through to each GGRN backend, except beforehand, the ggrn arg `prediction_timescale` will be outer-joined with it. So each backend will receive a dataframe similar to `predictions_metadata`, but with one more column called `prediction_timescale` and with N times as many rows where N is the number of timepoints in the user-specified input `prediction_timescale`. To maintain DRY, any representation of `prediction_timescale` outside `predictions_metadata` should be ignored by individual backends (or even better, not passed to them). 

### Evaluation

#### Limits of models and new considerations

In an ideal scenario, each method would make specific and literal predictions of the form, "Starting from cell type $A$ and timepoint $T_1$ in the training data, we alter the dosage of gene $G$ to a value of $E$, and then we run the model forward a number of steps $S$, yielding a predicted expression profile of $X$ at time $T_1+S$." Realistically, this will not happen for a few reasons. 

- Modeling limitations: 
    - Among the models we test, only PRESCIENT's timescale can be interpreted straightforwardly. scKINETICS and CellOracle's timescales are not calibrated, meaning it is unclear how long a single forward step of the model takes (but in my semi-expert opinion, it is probably a very short time, on the order of 6 hours). Dictys makes steady-state predictions (infinite time), but still uses cell type-specific GRN's, so in practice each prediction is confined to that GRN's cell type even if it's a developmental intermediate. How do we decide what concrete observations to compare each prediction against??
    - Some models, notably CellOracle and PRESCIENT, are not meant to be interpreted gene by gene. They advertise only the ability to predict low-dimensional summaries of cell state, like cluster labels or PCA embeddings. 
- Data limitations:
    - Timeseries and perturbation datasets usually have batch effects, making the quantitative expression levels not comparable. Sometimes the source of the effect is known: for example, in the definitive endoderm data, the timeseries data are from methanol-fixed cells and the perturb-seq from live cells. Other times, e.g. BETS, the source of the batch effect is not known ahead of time, but unsupervised analysis has raised red flags.
    - The post-knockdown level of each gene is not always known. It may be observed in a final cell state, but we don't know what it was at the intermediate timepoints.

#### Metrics to compute

Based on past work, here are some signals that we might still have success predicting.

- Short-term change in expression. Dictys uses this for model evaluation.
- Short-term change in PCA or other embeddings. CellOracle and scKINETICS focus on this.
- Short-term change in cell type proportions. CellOracle focuses on this.
- Long-term change in cell type proportions. PRESCIENT focuses on this.
- Cell type-specific TF activity: rank the TF's by the magnitude of the change predicted upon knockout, e.g. cosine similarity of velocity before and after KO. scKINETICS and CellOracle focus on this.

#### Matching observed and predicted expression

The methods in this section are implemented in `select_comparable_observed_and_predicted` in `evaluator.py`.

We previously assumed one prediction for every test-set sample. We no longer enjoy this luxury, and instead, we must deliberately select out fair pairings of predicted and observed samples. Specifically: 

- the implementation described above may return *trajectories* of predictions. For example, we may run CellOracle and make predictions at 1,2,3, and 10 time-steps, yielding 4 observations per starting state per perturbation. 
- to reduce the total computational burden, above we planned to return just one trajectory per starting state, not one per cell. This means the test data may contain many replicates per perturbation, but the predictions will not. This is again a problem for evaluation code that previously required a one-to-one match between predicted and observed samples. 

The new default behavior should be:

- For multiple predictions made after different numbers of time-steps, evaluate these separately, even if they are returned inside the same AnnData object. This will require new code inside of `evaluator.evaluateCausalModel` to iterate over values of `predictions.obs["prediction_timescale"]`.
- Compare predictions to test data within each cell type and timepoint, averaging together all test samples. This will require new code inside of `evaluator.evaluateCausalModel` to do the averaging.
- `evaluator.evaluateCausalModel` should still return a tuple of dataframes. The dataframes should have mostly the same columns. `is_timescale_strict` will be new. I don't remember if `timepoint`, `cell_type`, and `prediction_timescale` will be new or not. The rest should be the same. 

For a given starting timepoint, starting celltype, and prediction timescale, how do we select a sensible observation to test it against? For some methods, the simulation is moving forward in time. For most methods, the simulation is really just describing cell-state-specific perturbation effects with minimal to no progress in time. To represent this dichotomy, we will add a boolean parameter `does_simulation_progress`. If `True`, the evaluation will be PRESCIENT-style, so that `prediction_timescale + timepoint` in the prediction metadata is equal to `timepoint` in the observed expression data. If `False`, the evaluation will be CO-style, so that `timepoint` in the prediction metadata is equal to `timepoint` in the observed expression data. We will expose `does_simulation_progress` to the user through `metadata.json`. `does_simulation_progress` will default to `False`, unless the backend is prescient, autoregressive, or timeseries baseline, in which case it will default to `True`. This also affects the timepoints that the benchmarking software requests. In all cases, any variable called `timepoint` will mean the time at which the simulation **starts**.

The plan above hits one pitfall when KO's cause a block in differentiation, as in the endoderm data. In these cases, it would be more sensible to relax the requirement that `timepoint` in the prediction metadata be equal to `timepoint` in the observed expression data. In practice, a strong effect inhibiting differentiation at an intermediate timepoint could be selected as the most interesting hypothesis and compared directly to the observed data. To allow for this, we will add a boolean flag called `is_timescale_strict`. If true, we will enforce that the takedown timepoint be equal between observed and predicted perturbed profiles. If false, we will select the `"cell_type", "timepoint"` combo with the largest effect and compare that against the observed expression for the same perturbation. For fairness, we will enable this option even for methods like PRESCIENT that are very literal about timescale interpretation. By default, we will conduct and return evaluations with both possible settings of `is_timescale_strict`.

Long-term simulation creates problems for cell type labels too. In complex timeseries such as the Saunders whole zebrafish embryo scRNA, it is hard to name the precursor that will give rise to a specified terminal cell type 48 or 72 hours later. In other words, there is a difficult lineage tracing problem embedded inside the causal GRN benchmarking problem. GGRN and PRESCIENT offer various solutions (see discussion of `matching_method` in the [GGRN exposition](https://github.com/ekernf01/ggrn/blob/main/docs/GGRN.md)), but these are approximations for purposes of training specific models. For ground truth, we need an approach that is neutral and data-driven. So the current plan is to make predictions from every available starting state, then before evaluation, reassign cell type labels using a classifier. 

One more thing requires some clarification: what exactly is a "cell type" for purposes of method evaluation? 

- A549 and FANTOM4: let's not try to classify these into cell types -- all train data, test data, and predictions should be labeled as just a single cell type. 
- Hematopoiesis: We can use "cell_type" in a similar way to the CellOracle demos on these data -- by contrast to the cell line data, there is a lot of diversity and differentiation here.
- Zebrafish: We can just use the original authors' cell type annotation. 
- Endoderm: this is a tricky case because the knockdowns yield one cell type that doesn't occur in the timeseries data and two others that don't match exactly. See [fig. 2c](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6525305/). Currently the data have "cell_type" set to "endoderm_differentiation", but I think I should rename it to "PSC", "primitive streak", "endoderm", or "ectopic" in both the training and test data. This is an accurate summary of the biology, but it raises a problem: methods using cell type specific networks will not have an "ectopic" network available because that cell type is by definition absent from the training data. This is a brutally hard test case, so it should be low priority. For now, let's substitute in all cell types from the training data, get the predictions, but leave the evaluation code as it is. We will probably want to do manual evaluation since it's a small number of perturbations and a difficult edge case. As of June 26, we would need to modify construction of `predictions_metadata` in `do_one_experiment.py`. 

#### Baseline expression 

When evaluating fold changes, we compute them against some baseline expression. What should this be? (Note: I'm not talking about a baseline predictive method. I'm talking about pre-perturbation expression.)

- For train-test splits of a single perturbset, usually the controls are all in the training data. The same baseline can be used for the training and test data, and it needs to be extracted from the training data. 
- For timeseries-versus-perturbseq splits, the baseline should be different between predicted and test data, because batch effects usually make the timeseries and perturb-seq not directly comparable. For the test data, it should be a **test-set** control sample from the same timepoint and cell type. For the predictions, it should be a **training-set** control sample or a **prediction under no perturbations** from the same timepoint and cell type.

#### Baseline predictive methods

For the previous part of the project, an important component of the results came from using non-informative baselines, e.g. predicting the median of the training data or using the empty network for network-based feature selection. What are some simple or non-informative baselines for prediction of a knockout of gene $G$ at time $t+1$ starting the simulation from time $t$?

- return $X_t$ (no development)
    - We could probably implement this via the existing autoregressive backend. It roughly does X = X_0 + BX_0, and we can just set B=0 via an empty network, a high penalty param, or a rank-0 matrix. But does any of these options not choke the software??? This is not yet implemented and I'm not sure if it's worth doing.
- return $X_{t+1}$ (development proceeds and is not affected by KO)
    - This is now implemented via the ggrn `timeseries_baseline` docker backend.
    - This is a really sensible and important baseline. 