"""evaluator.py is a collection of functions for testing predictions about expression fold change.
"""
from joblib import Parallel, delayed, cpu_count
from joblib.parallel import parallel_config
import numpy as np
import pandas as pd
import anndata
from scipy.stats import spearmanr as spearmanr
from scipy.stats import rankdata as rank
import os 
import re
import altair as alt
import pereggrn.experimenter as experimenter
from  scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from typing import Tuple, Dict, List
import gc

def test_targets_vs_non_targets( predicted, observed, baseline_predicted, baseline_observed ): 
    predicted = np.squeeze(np.array(predicted))
    observed = np.squeeze(np.array(observed))
    baseline_predicted = np.squeeze(np.array(baseline_predicted))
    baseline_observed = np.squeeze(np.array(baseline_observed))
    targets_positive = np.sign(np.round( predicted - baseline_predicted, 2))== 1
    targets_negative = np.sign(np.round( predicted - baseline_predicted, 2))==-1
    non_targets      = np.sign(np.round( predicted - baseline_predicted, 2))== 0
    fc_observed = observed - baseline_observed
    if targets_positive.any() and targets_negative.any() and non_targets.any():
        return f_oneway(
            fc_observed[targets_positive],
            fc_observed[targets_negative],
            fc_observed[non_targets],
        ).pvalue
    elif non_targets.any() and targets_negative.any():
        return f_oneway(
            fc_observed[targets_negative],
            fc_observed[non_targets],
        ).pvalue    
    elif targets_positive.any() and targets_negative.any():
        return f_oneway(
            fc_observed[targets_positive],
            fc_observed[targets_negative],
        ).pvalue
    elif targets_positive.any() and non_targets.any():
        return f_oneway(
            fc_observed[targets_positive],
            fc_observed[non_targets],
        ).pvalue
    else:
        return np.nan
    

def fc_targets_vs_non_targets( predicted, observed, baseline_predicted, baseline_observed ): 
    predicted = np.squeeze(np.array(predicted))
    observed = np.squeeze(np.array(observed))
    baseline_observed = np.squeeze(np.array(baseline_observed))
    baseline_predicted = np.squeeze(np.array(baseline_predicted))
    targets_positive = np.sign(np.round( predicted - baseline_predicted, 2))== 1
    targets_negative = np.sign(np.round( predicted - baseline_predicted, 2))==-1
    non_targets      = np.sign(np.round( predicted - baseline_predicted, 2))== 0
    fc_observed = observed - baseline_observed
    if targets_positive.any() and targets_negative.any() and non_targets.any():
        return fc_observed[targets_positive].mean() - fc_observed[targets_negative].mean()
    elif non_targets.any() and targets_negative.any():
        return fc_observed[non_targets].mean() - fc_observed[targets_negative].mean()
    elif targets_positive.any() and targets_negative.any():
        return fc_observed[targets_positive].mean() - fc_observed[targets_negative].mean()
    elif targets_positive.any() and non_targets.any():
        return fc_observed[targets_positive].mean() - fc_observed[non_targets].mean()
    else:
        return np.nan

METRICS = {
    "spearman":                     lambda predicted, observed, baseline_predicted, baseline_observed: [x for x in spearmanr(observed - baseline_observed,   predicted - baseline_predicted)][0],
    "mae":                          lambda predicted, observed, baseline_predicted, baseline_observed: np.abs               (observed - baseline_observed - (predicted - baseline_predicted)).mean(),
    "mse":                          lambda predicted, observed, baseline_predicted, baseline_observed: np.linalg.norm       (observed - baseline_observed - (predicted - baseline_predicted))**2,
    "mse_top_20":                   lambda predicted, observed, baseline_predicted, baseline_observed: mse_top_n(predicted, observed, baseline_predicted, baseline_observed, n=20),
    "mse_top_100":                  lambda predicted, observed, baseline_predicted, baseline_observed: mse_top_n(predicted, observed, baseline_predicted, baseline_observed, n=100),
    "mse_top_200":                  lambda predicted, observed, baseline_predicted, baseline_observed: mse_top_n(predicted, observed, baseline_predicted, baseline_observed, n=200),
    "proportion_correct_direction": lambda predicted, observed, baseline_predicted, baseline_observed: np.mean(np.sign(observed - baseline_observed) == np.sign(predicted - baseline_predicted)),
    "pvalue_effect_direction":      lambda predicted, observed, baseline_predicted, baseline_observed: chi2_contingency(
        observed = pd.crosstab(
            np.sign(np.round( observed - baseline_observed, 2)),
            np.sign(np.round(predicted - baseline_predicted, 2))
        )
    ).pvalue,
    "pvalue_targets_vs_non_targets":  test_targets_vs_non_targets,
    "fc_targets_vs_non_targets": fc_targets_vs_non_targets,
}

def mse_top_n(predicted, observed, baseline_predicted, baseline_observed, n):
    top_n = rank(-np.abs(observed - baseline_observed)) <= n
    return np.linalg.norm((observed - baseline_observed - (predicted - baseline_predicted))[top_n]) ** 2

def makeMainPlots(
    evaluationPerPert: pd.DataFrame, 
    evaluationPerTarget: pd.DataFrame, 
    outputs: str, 
    factor_varied:str, 
    facet_by: str = None, 
    color_by: str = None, 
    metrics = [
        'spearman', 'mse', 'mae',
        "mse_top_20", "mse_top_100", "mse_top_200",
    ]
    ):
    """Redo the main plots summarizing an experiment.
    Args:
        evaluationPerPert (pd.DataFrame)
        evaluationPerTarget (pd.DataFrame)
        factor_varied (str): Plots are automatically colored based on this column of "evaluationPerPert". 
        facet_by (str): Plots are automatically stratified based on this column of "evaluationPerPert". 
        outputs (str): folder to save plots in
        metrics: How to measure performance. 
    """
    # Sometimes the index is too complex for Altair to handle correctly (tuples)
    evaluationPerPert = evaluationPerPert.copy()
    try:
        evaluationPerPert.index = [p[1] for p in evaluationPerPert.index]
    except IndexError:
        pass
    vlnplot = {}
    _ = alt.data_transformers.disable_max_rows()
    if color_by is not None:
        evaluationPerPert[factor_varied + " "] = [str(a) + str(b) for a,b in zip(evaluationPerPert[factor_varied], evaluationPerPert[color_by])]
        group_mean_by = [factor_varied + " "]
    else:
        group_mean_by = [factor_varied]
    if facet_by is not None:
        group_mean_by.append(facet_by)
    for metric in metrics:
        means = evaluationPerPert.groupby(group_mean_by, as_index=False)[[metric]].mean()
        vlnplot[metric] = alt.Chart(
                data = evaluationPerPert, 
                title = f"{metric} (predicted log fold change vs observed)"
            ).mark_boxplot(extent='min-max')
        # Faceting fights with layering, so skip the means if faceting.
        if facet_by is None:
            vlnplot[metric] = vlnplot[metric] + alt.Chart(data = means).mark_point(color="black")
        if color_by is not None:
            vlnplot[metric]=vlnplot[metric].encode(
                y=alt.Y(f'{metric}:Q'),
                color=color_by + ':N',
                x=alt.X(
                    factor_varied + " " + ':N'
                )
            ).properties(
                width=400,
                height=400
            )
        else:
            vlnplot[metric] = vlnplot[metric].encode(
                y=alt.Y(f'{metric}:Q'),
                x=alt.X(
                    factor_varied + ':N'
                )
            ).properties(
                width=400,
                height=400
            )
        if facet_by is not None:
            vlnplot[metric] = vlnplot[metric].facet(
                facet_by + ':N',
                columns=int(np.ceil(np.sqrt(len(evaluationPerPert[facet_by].unique())))), 
            )
        try:
            vlnplot[metric].save(f'{outputs}/{metric}.svg')
        except Exception as e:
            print(f"Got error {repr(e)} during svg saving; trying instead with html and interactive html.", flush = True)
            vlnplot[metric].save(f'{outputs}/{metric}.html')    
    return vlnplot

def addGeneMetadata(
        df: pd.DataFrame, 
        adata: anndata.AnnData,
        adata_test: anndata.AnnData,
        genes_considered_as: str,
        path_to_accessory_data: str,
    ) -> pd.DataFrame:
    """Add metadata related to evo conservation and network connectivity

    Args:
        df (pd.DataFrame): Gene names and associated performance metrics
        adata (anndata.AnnData): training expression data
        adata_test (anndata.AnnData): test-set expression data
        genes_considered_as (str): "targets" or "perturbations"

    Returns:
        pd.DataFrame: df with additional columns describing evo conservation and network connectivity
    """
    # Measures derived from the test data, e.g. effect size
    if genes_considered_as == "targets":
        df["gene"] = df.index
    else: # genes_considered_as == "perturbations"
        df["gene"] = df["perturbation"]
        perturbation_characteristics = {
            'fraction_missing',
            'logFC',
            'spearmanCorr',
            'pearsonCorr',
            'logFCNorm2',
        }.intersection(adata_test.obs.columns)
        df = pd.merge(
            adata_test.obs.loc[:,["perturbation"] + list(perturbation_characteristics)],
            df.copy(),
            how = "outer", # Will deal with missing info later
            left_on="perturbation", 
            right_on="gene"
        )   

    # Measures derived from the training data, e.g. overdispersion
    expression_characteristics = [
        'highly_variable', 'highly_variable_rank', 'means',
        'variances', 'variances_norm'
    ]
    expression_characteristics = [e for e in expression_characteristics if e in adata.var.columns]
    if any(not x in df.columns for x in expression_characteristics):
        df = pd.merge(
            adata.var[expression_characteristics],
            df.copy(),
            how = "outer", # This yields missing values. Will deal with that later
            left_index=True, 
            right_on="gene")
    # Proteoform diversity information is not yet used because it would be hard to summarize this into a numeric measure of complexity.
    # But this code may be useful if we aim to continue that work later on.
    proteoform_diversity = pd.read_csv(os.path.join(
            path_to_accessory_data, 
            "uniprot-compressed_true_download_true_fields_accession_2Cid_2Cprotei-2023.02.02-15.27.12.44.tsv.gz"
        ), 
        sep = "\t"
    )
    proteoform_diversity.head()
    proteoform_diversity_summary = pd.DataFrame(
        {
            "is_glycosylated": ~proteoform_diversity["Glycosylation"].isnull(),
            "has_ptm": ~proteoform_diversity["Post-translational modification"].isnull(),
        },
        index = proteoform_diversity.index,
    )
    proteoform_diversity_characteristics = proteoform_diversity_summary.columns.copy()

    # measures of evolutionary constraint 
    evolutionary_characteristics = ["pLI"]
    evolutionary_constraint = pd.read_csv(os.path.join(
            path_to_accessory_data, 
            "forweb_cleaned_exac_r03_march16_z_data_pLI_CNV-final.txt.gz"
        ), 
        sep = "\t"
    )

    evolutionary_constraint = evolutionary_constraint.groupby("gene").agg(func = max)
    if any(not x in df.columns for x in evolutionary_characteristics):
        df = pd.merge(
            evolutionary_constraint,
            df.copy(),
            how = "outer", # This yields missing values. Will deal with that later
            left_on="gene", 
            right_on="gene")
    
    # measures of connectedness
    degree = pd.read_csv(os.path.join(
            path_to_accessory_data, 
            "degree_info.csv.gz"
        )
    )
    degree = degree.rename({"Unnamed: 0":"gene"}, axis = 1)
    degree["gene"] = [str(g).upper() for g in degree["gene"]]
    degree = degree.pivot_table(
        index=['gene'], 
        values=['in-degree', 'out-degree'], 
        columns=['network']
    )
    degree.fillna(0)
    degree.columns = ['_'.join(col) for col in degree.columns.values]
    degree_characteristics = list(degree.columns)
    if any(not x in df.columns for x in degree_characteristics):
        df = pd.merge(
            degree,
            df.copy(),
            how = "outer", # This yields missing values. Will deal with that later
            left_on="gene", 
            right_on="gene")
    try:
        df.reset_index(inplace=True)
    except:
        pass
    types_of_gene_data = {
        "evolutionary_characteristics":evolutionary_characteristics,
        "expression_characteristics": expression_characteristics, 
        "degree_characteristics": degree_characteristics,
    }
    if genes_considered_as == "perturbations":
        types_of_gene_data["perturbation_characteristics"] = perturbation_characteristics
    
    # Remove missing values from outer joins.
    # These are genes where we have various annotations, but they are not actually
    # perturbed or not actually measured on the test set.
    df = df.loc[df["mae"].notnull(), :]
    return df, types_of_gene_data

def plotOneTargetGene(gene: str, 
                      outputs: str, 
                      conditions: pd.DataFrame, 
                      factor_varied: str,
                      train_data: anndata.AnnData, 
                      heldout_data: anndata.AnnData, 
                      fitted_values: anndata.AnnData, 
                      predictions: anndata.AnnData) -> None:
    """For one gene, plot predicted + observed logfc for train + test.

    Args:
        gene (str): gene name (usually the HGNC symbol)
        outputs (str): where to save the plots
        conditions (pd.DataFrame): Metadata from conditions.csv
        factor_varied (str): what to use as the x axis in the plot
        train_data (anndata.AnnData): training expression
        heldout_data (anndata.AnnData): test-set expression
        fitted_values (anndata.AnnData): predictions about perturbations in the training set
        predictions (anndata.AnnData): predictions about perturbations in the test set
    """
    expression = {
        e:pd.DataFrame({
            "index": [i for i in range(
                fitted_values[e][:,gene].shape[0] + 
                predictions[e][:,gene].shape[0]
            )],
            "experiment": e,
            "observed": np.concatenate([
                safe_squeeze(train_data[e][:,gene].X), 
                safe_squeeze(heldout_data[e][:,gene].X), 
            ]), 
            "predicted": np.concatenate([
                safe_squeeze(fitted_values[e][:,gene].X), 
                safe_squeeze(predictions[e][:,gene].X), 
            ]), 
            "is_trainset": np.concatenate([
                np.ones (fitted_values[e][:,gene].shape[0]), 
                np.zeros(  predictions[e][:,gene].shape[0]), 
            ]), 
        }) for e in predictions.keys() 
    }
    expression = pd.concat(expression)
    expression = expression.reset_index()
    expression = expression.merge(conditions, left_on="experiment", right_index=True)
    os.makedirs(os.path.join(outputs), exist_ok=True)
    alt.Chart(data=expression).mark_point().encode(
        x = "observed:Q",y = "predicted:Q", color = "is_trainset:N"
    ).properties(
        title=gene
    ).facet(
        facet = factor_varied, 
        columns=3,
    ).save(os.path.join(outputs, gene + ".svg"))
    return   

def postprocessEvaluations(evaluations: pd.DataFrame, 
                           conditions: pd.DataFrame)-> pd.DataFrame:
    """Add condition metadata to eval results and fix formatting.

    Args:
        evaluations (pd.DataFrame): evaluation results for each test-set observation
        conditions (pd.DataFrame): metadata from conditions.csv

    Returns:
        pd.DataFrame: evaluation results with experimental conditions
    """
    evaluations   = pd.concat(evaluations)
    evaluations   = evaluations.merge(conditions,   how = "left", right_index = True, left_on = "index")
    evaluations   = pd.DataFrame(evaluations.to_dict()) # This cleans up weird data types
    return evaluations

def evaluateCausalModel(
    get_current_data_split:callable, 
    predicted_expression: dict,
    is_test_set: bool,
    conditions: pd.DataFrame, 
    outputs: str, 
    classifier_labels = None,
    do_scatterplots = True, 
    path_to_accessory_data: str = "../accessory_data",
    do_parallel: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compile plots and tables comparing heldout data and predictions for same. 

    Args:
        get_current_data_split: function to retrieve tuple of anndatas (train, test)
        predicted_expression: dict with keys equal to the index in "conditions" and values being anndata objects. 
        is_test_set: True if the predicted_expression is on the test set and False if predicted_expression is on the training data.
            This is just used to select the right observations to compare to the predictions.
        conditions (pd.DataFrame): Metadata for the different combinations used in this experiment. 
        outputs (String): Saves output here.
        classifier_labels (String): Column in conditions to use as the target for the classifier.
        do_scatterplots (bool): Make scatterplots of observed vs predicted expression.
        path_to_accessory_data (str): We use this to add gene metadata on LoF intolerance.
        do_parallel (bool): Use joblib to parallelize the evaluation across perturbations. Recommended unless you are debugging (it ruins tracebacks).
    """
    evaluationPerPert = {}
    evaluationPerTarget = {}
    evaluations  = []
    for i in predicted_expression.keys():
        perturbed_expression_data_train_i, perturbed_expression_data_heldout_i = get_current_data_split(i)
        projector = PCA(n_components = 20)
        try:
            projector.fit(perturbed_expression_data_train_i.X.toarray())
        except AttributeError:
            projector.fit(perturbed_expression_data_train_i.X)        
        all_test_data = perturbed_expression_data_heldout_i if is_test_set else perturbed_expression_data_train_i
        
        evaluations = {}
        if "prediction_timescale" not in predicted_expression[i].obs.columns:
            predicted_expression[i].obs["prediction_timescale"] = conditions.loc[i, "prediction_timescale"]
        timescales = predicted_expression[i].obs["prediction_timescale"].unique()
        predicted_expression[i] = predicted_expression[i].to_memory(copy = True)
        predicted_expression[i] = predicted_expression[i][pd.notnull(predicted_expression[i].X.sum(1)), :]
        for prediction_timescale in timescales:
            if (conditions.loc[i, "type_of_split"] == "timeseries"):
                # For timeseries-versus-perturbseq splits, baseline and observed-to-predicted matching are more complicated. See `docs/timeseries_prediction.md` for details.
                # this returns anndatas in the order OBSERVED, PREDICTED
                current_heldout, predicted_expression_it = select_comparable_observed_and_predicted(
                    conditions, 
                    predicted_expression[i], 
                    all_test_data, 
                    i,
                    # I don't care if this is somewhat redundant with the classifier used below. We need both even if not elegant.
                    classifier = experimenter.train_classifier(perturbed_expression_data_train_i, target_key = "cell_type")
                )   
                # The sensible baseline differs between predicted and test data. 
                # For the test data, it should be a **test-set** control sample from the same timepoint and cell type. 
                # For the predictions, it should be a **prediction under no perturbations** from the same timepoint and cell type. 
                # Because the upstream code selects perturbations to predict from the test set, the names of the controls should match the heldout data.
                baseline_observed = current_heldout.copy()[current_heldout.obs["is_control"], :]
                baseline_predicted = predicted_expression_it[ predicted_expression_it.obs["is_control"], : ].copy()
            else:
                current_heldout = all_test_data
                predicted_expression_it = predicted_expression[i]
                # For train-test splits of a single perturbset, the controls are all in the training data. 
                # The same baseline can be used for the training and test data, and it needs to be extracted from the training data. 
                baseline_observed  = perturbed_expression_data_train_i[[bool(b) for b in perturbed_expression_data_train_i.obs["is_control"]], :]
                baseline_predicted = baseline_observed.copy()

            classifier_labels = "cell_type" if (conditions.loc[i, "type_of_split"]=="timeseries") else None # If you pass None, it will look for "louvain" or give up.
            evaluations[prediction_timescale] = evaluateOnePrediction(
                expression = current_heldout,
                predictedExpression = predicted_expression_it,
                baseline_observed = baseline_observed,
                baseline_predicted = baseline_predicted,
                doPlots=do_scatterplots,
                outputs = outputs,
                experiment_name = i,
                classifier = experimenter.train_classifier(perturbed_expression_data_train_i, target_key = classifier_labels),
                projector = projector,
                do_parallel=do_parallel,
                is_timeseries = (conditions.loc[i, "type_of_split"] == "timeseries"),
            )
            # Add detail on characteristics of each gene that might make itq more predictable
            evaluations[prediction_timescale][0], _ = addGeneMetadata(evaluations[prediction_timescale][0], genes_considered_as="perturbations", adata=perturbed_expression_data_train_i, adata_test=perturbed_expression_data_heldout_i, path_to_accessory_data=path_to_accessory_data)
            evaluations[prediction_timescale][1], _ = addGeneMetadata(evaluations[prediction_timescale][1], genes_considered_as="targets"      , adata=perturbed_expression_data_train_i, adata_test=perturbed_expression_data_heldout_i, path_to_accessory_data=path_to_accessory_data)
            evaluations[prediction_timescale][0]["index"] = i
            evaluations[prediction_timescale][1]["index"] = i
            evaluations[prediction_timescale][0]["prediction_timescale"] = prediction_timescale
            evaluations[prediction_timescale][1]["prediction_timescale"] = prediction_timescale
            
        evaluationPerPert  [i] = pd.concat([evaluations[t][0] for t in timescales])
        evaluationPerTarget[i] = pd.concat([evaluations[t][1] for t in timescales])
        assert "prediction_timescale" in evaluationPerPert[i].columns
        assert "prediction_timescale" in evaluationPerTarget[i].columns
        predicted_expression[i] = None
        gc.collect()
    # Concatenate and add some extra info
    # postprocessEvaluations wants a list of datafrmaes with one dataframe per row in conditions
    try: 
        del conditions["prediction_timescale"] # this is now redundant, and if not deleted, it will mess up a merge in postprocessEvaluations
    except KeyError:
        pass
    evaluationPerPert   = postprocessEvaluations(evaluationPerPert, conditions)
    evaluationPerTarget = postprocessEvaluations(evaluationPerTarget, conditions)
    assert "prediction_timescale" in evaluationPerPert.columns
    assert "prediction_timescale" in evaluationPerTarget.columns
    return evaluationPerPert, evaluationPerTarget

def select_comparable_observed_and_predicted(
    conditions: pd.DataFrame, 
    predictions: anndata.AnnData, 
    perturbed_expression_data_heldout_i: anndata.AnnData, 
    i: int, 
    classifier,
) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """Select a set of predictions that are comparable to the test data, and aggregate the test data within each combination of
    perturbed gene, timepoint, and cell_type. See docs/timeseries_prediction.md for details.
    This function should NOT be run unless type_of_split is "timeseries".

    Args:
        conditions (pd.DataFrame): all experimental conditions
        predictions (anndata.AnnData): predicted expression
        perturbed_expression_data_heldout_i (anndata.AnnData): the heldout data
        i (int): which condition you are currently preparing to evaluate
        classifier: for methods like PRESCIENT where the simulation can last a long time, we may need to update the "cell_type" labels. 
            This should be an sklearn classifier with a .predict() method. It should accept all genes as features.

    Returns:
        Tuple[anndata.AnnData, anndata.AnnData]: the observed test data and the predictions, with a one-to-one match between them.
    """
    # maintain backwards compatibility, this allows these fields to be missing.
    if not "timepoint" in perturbed_expression_data_heldout_i.obs.columns: 
        perturbed_expression_data_heldout_i.obs["timepoint"] = 0
    if not "cell_type" in perturbed_expression_data_heldout_i.obs.columns: 
        perturbed_expression_data_heldout_i.obs["cell_type"] = 0
    # For timeseries datasets, we just return a single prediction for each timepoint and cell_type, not one for each test set cell.
    test_data = experimenter.averageWithinPerturbation(perturbed_expression_data_heldout_i, ["timepoint", "cell_type"])
    # Now we will select predictions for each combo of timepoint, cell_type, and perturbation in the heldout data.
    predictions.obs["takedown_timepoint"] = predictions.obs["timepoint"]
    if conditions.loc[i, "does_simulation_progress"]:
        predictions.obs.loc[:, "takedown_timepoint"] += predictions.obs.loc[:, "prediction_timescale"]
        predictions.obs.loc[:, "cell_type"] = classifier.predict(predictions.X)

    test_data.obs["observed_index"] = test_data.obs.index.copy()
    predictions.obs["predicted_index"] = predictions.obs.index.copy()
    
    # this merge is (john mulaney voice) sensitive about data types
    test_data.obs["perturbation"] = test_data.obs["perturbation"].astype("str")
    predictions.obs["perturbation"] = predictions.obs["perturbation"].astype("str")
    test_data.obs["cell_type"] = test_data.obs["cell_type"].astype("str")
    predictions.obs["cell_type"] = predictions.obs["cell_type"].astype("str")
    test_data.obs["timepoint"] = test_data.obs["timepoint"].astype("Int64")
    predictions.obs["takedown_timepoint"] = predictions.obs["takedown_timepoint"].astype("Int64")
    matched_predictions = pd.merge(
        test_data.obs[              [         'timepoint',          "cell_type", 'perturbation', "observed_index"]], 
        predictions.obs[['takedown_timepoint',          "cell_type", 'perturbation', "predicted_index"]], 
        left_on =['timepoint',          "cell_type", 'perturbation'],
        right_on=['takedown_timepoint', "cell_type", 'perturbation'],
        how='inner', 
    )
    # TODO: this will fail on multi-gene perturbations. Fix it or fill in .obs["is_control"] upstream.
    if "is_control" not in predictions.obs.columns:
        predictions.obs["is_control"] = pd.isnull(predictions.obs["expression_level_after_perturbation"]) | ~predictions.obs["perturbation"].isin(predictions.var_names)
    assert "is_control" in predictions.obs.columns
    assert "is_control" in test_data.obs.columns
    new_index = [str(j) for j in range(matched_predictions.shape[0])]
    observed = test_data[matched_predictions["observed_index"]]
    predicted = predictions[matched_predictions["predicted_index"], :]
    predicted.obs_names = new_index
    observed.obs_names = new_index
    return observed, predicted

def safe_squeeze(X):
    """Squeeze a matrix when you don't know if it's sparse-format or not.

    Args:
        X (np.matrix or scipy.sparse.csr_matrix): _description_

    Returns:
        np.array: 1-d version of the input
    """
    try:
        X = X.toarray()
    except:
        pass
    try:
        X = np.asarray(X)
    except:
        pass
    X = X.squeeze()
    assert len(X.shape)==1, "squeeze failed -- is expression stored in a weird type of array other than numpy matrix/array/memmap, anndata view, or scipy csr sparse?"
    return X

def evaluate_per_target(i: int, target: str, expression, predictedExpression):
    """Evaluate performance on a single target gene.

    Args:
        i (int): index of target gene to check
        target (str): name of target gene
        expression (np or scipy matrix): true expression or logfc
        predictedExpression (np or scipy matrix): predicted expression or logfc

    Returns:
        tuple: target, std_dev, mae, mse where target is the gene name, std_dev is the standard deviation of the 
            predictions (to check if they are constant), and mae and mse are mean absolute or squared error
    """
    observed  = safe_squeeze(expression[:, i])
    predicted = safe_squeeze(predictedExpression[:, i])
    std_dev = np.std(predicted)
    mae = np.abs(observed - predicted).sum().copy()
    mse = np.linalg.norm(observed - predicted) ** 2
    return target, std_dev, mae, mse

def evaluate_across_targets(expression: anndata.AnnData, predictedExpression: anndata.AnnData) -> pd.DataFrame:
    """Evaluate performance for each target gene.

    Args:
        expression (anndata.AnnData): actual expression or logfc
        predictedExpression (anndata.AnnData): predicted expression or logfc

    Returns:
        pd.DataFrame: _description_
    """
    targets = predictedExpression.var.index
    predictedExpression = predictedExpression.to_memory()
    with parallel_config(temp_folder='/tmp'):
        results = Parallel(n_jobs=cpu_count()-1)(delayed(evaluate_per_target)(i, target, expression.X, predictedExpression.X) for i,target in enumerate(targets))
    metrics_per_target = pd.DataFrame(results, columns=["target", "standard_deviation", "mae", "mse"]).set_index("target")
    return metrics_per_target

def evaluate_per_pert(
        pert: str,
        all_perts: pd.Series,
        expression: np.matrix, 
        predictedExpression: np.matrix,
        baseline_predicted: np.matrix, 
        baseline_observed: np.matrix, 
        classifier = None, 
        projector = None,
    ) -> pd.DataFrame:
    """Calculate evaluation metrics for one perturbation. 

    Args:
        pert (str): name(s) of perturbed gene(s)
        all_perts (pd.Series): name(s) of perturbed gene(s), one per sample in predictedExpression
        expression (np.matrix): actual expression, log1p-scale. We use a matrix, not an AnnData, for fast parallelism via memory sharing.
        predictedExpression (np.matrix): predicted expression, log1p-scale
        baseline (np.matrix): baseline expression, log1p-scale
        classifier (optional): None or sklearn classifier to judge results by cell type label accuracy
        projector (optional): None or sklearn PCA object to project expression into a lower-dimensional space

    Returns:
        pd.DataFrame: Evaluation results for each perturbation
    """
    i = all_perts==pert
    predicted = safe_squeeze(predictedExpression[i, :].mean(axis=0))
    observed  = safe_squeeze(         expression[i, :].mean(axis=0))
    assert observed.shape[0] == expression.shape[1], f"For perturbation {pert}, observed and predicted are different shapes."
    def is_constant(x):
        return np.std(x) < 1e-12
    if np.isnan(predicted).any() or is_constant(predicted - baseline_predicted) or is_constant(observed - baseline_observed):
        return pd.DataFrame({m:np.nan for m in METRICS.keys()}, index = [pert])
    else:
        results = {k:m(predicted, observed, baseline_predicted, baseline_observed) for k,m in METRICS.items()}
        results["cell_type_correct"] = np.nan
        if classifier is not None:
            class_observed = classifier.predict(np.reshape(observed, (1, -1)))[0]
            class_predicted = classifier.predict(np.reshape(predicted, (1, -1)))[0]
            results["cell_type_correct"] = 1.0 * (class_observed == class_predicted)
        results["distance_in_pca"] = np.nan
        if projector is not None:
            results["distance_in_pca"] = np.linalg.norm(projector.transform(observed.reshape(1, -1)) - projector.transform(predicted.reshape(1, -1)))**2
        return pd.DataFrame(results, index = [pert])

def evaluate_across_perts(expression: anndata.AnnData, 
                          predictedExpression: anndata.AnnData, 
                          baseline_predicted: anndata.AnnData, 
                          baseline_observed: anndata.AnnData, 
                          experiment_name: str, 
                          classifier = None, 
                          projector = None,
                          do_careful_checks: bool=False, 
                          do_parallel: bool = True) -> pd.DataFrame:
    """Evaluate performance for each perturbation.

    Args:
        expression (anndata.AnnData): actual expression, log1p-scale
        predictedExpression (anndata.AnnData): predicted expression, log1p-scale
        baseline_predicted, baseline_observed (anndata.AnnData): baseline expression, log1p-scale
        experiment_name (str): name of the experiment
        classifier (optional): None or sklearn classifier to judge results by cell type label instead of logfc
        projector (optional): None or sklearn PCA object to project expression into a lower-dimensional space
        do_careful_checks (bool, optional): ensure that perturbation and dose match between observed
            and predicted expression. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    perts = predictedExpression.obs["perturbation"].unique()
    predictedExpression = predictedExpression.to_memory()
    if do_careful_checks:
        elap = "expression_level_after_perturbation"
        predictedExpression.obs[elap] = pd.to_numeric(predictedExpression.obs[elap], errors='coerce')
        expression.obs[elap] = pd.to_numeric(expression.obs[elap], errors='coerce')
        expression.obs[         "perturbation"] = expression.obs[         "perturbation"].astype(str)
        predictedExpression.obs["perturbation"] = predictedExpression.obs["perturbation"].astype(str)
        for c in ["perturbation", elap]:
            if not all(
                expression.obs.loc         [:, c].fillna(0) == 
                predictedExpression.obs.loc[:, c].fillna(0)
            ):
                print("Observed:")
                print(expression.obs.loc         [:, c].fillna(0).value_counts)
                print("Predicted:")
                print(predictedExpression.obs.loc[:, c].fillna(0).value_counts)
                raise ValueError(f"Expression and predicted expression have mismatched '{c}' metadata in experiment {experiment_name}. Check stdout for summary statistics.")
    if do_parallel:
        with parallel_config(temp_folder='/tmp', backend='threading'):
            results = Parallel(n_jobs=cpu_count())(
                delayed(evaluate_per_pert)(pert, expression.obs["perturbation"], expression.X, predictedExpression.X, baseline_predicted, baseline_observed, classifier, projector) 
                for pert in perts
            )
    else:
        results = [
            evaluate_per_pert(pert, expression.obs["perturbation"], expression.X, predictedExpression.X, baseline_predicted, baseline_observed, classifier, projector) 
            for pert in perts
        ]

    results = pd.concat([r for r in results if type(r) == pd.DataFrame])
    return results

def evaluateOnePrediction(
    expression: anndata.AnnData, 
    predictedExpression: anndata.AnnData, 
    baseline_predicted: anndata.AnnData, 
    baseline_observed: anndata.AnnData, 
    outputs,
    experiment_name: str,
    doPlots=False, 
    classifier=None, 
    projector=None,
    do_careful_checks = True, 
    do_parallel: bool = True, 
    is_timeseries: bool = False
):
    '''Compare observed against predicted, for expression, fold-change, or cell type.

            Parameters:
                    expression (AnnData): 
                        the observed expression post-perturbation (log-scale in expression.X). 
                    predictedExpression (AnnData): 
                        the cellOracle prediction (log-scale). Elements of predictedExpression.X may be np.nan for 
                        missing predictions, often one gene missing from all samples or one sample missing for all genes.
                        predictedExpression.obs must contain columns "perturbation" (symbol of targeted gene) 
                        and "expression_level_after_perturbation" (e.g. 0 for knockouts). 
                    baseline_predicted, baseline_observed (AnnData): 
                        control expression level (log-scale)
                    outputs (str): Folder to save output in
                    classifier (sklearn classifier): 
                        Random forest or other sklearn classifier to assign cell type to predicted expression profiles. 
                        Must have a predict() method capable of taking a value from expression or predictedExpression and returning a single class label. 
                    projector (sklearn PCA):
                        PCA or other sklearn dimension reduction object to project expression into a lower-dimensional space.
                    doPlots (bool): Make a scatterplot showing observed vs predicted, one dot per gene. 
                    do_careful_checks (bool): check gene name and expression level associated with each perturbation.
                        They must match between expression and predictionExpression.
                    do_parallel (bool): use joblib to parallelize the evaluation across perturbations.
                    is_timeseries (bool): for timeseries data we expect a different shape for observed and predicted. The default behavior is to compare
                        predictions to test data within each cell type and timepoint, averaging together all test samples. Also to evaluate predictions 
                        after different numbers of time-steps separately, even if multiple time-steps are returned inside the same AnnData object. 

            Returns:
                    Pandas DataFrame with Spearman correlation between predicted and observed 
                    log fold change over control.
    '''
    "log fold change using Spearman correlation and (optionally) cell fate classification."""
    if not expression.X.shape == predictedExpression.X.shape:
        raise ValueError(f"expression shape is {expression.X.shape} and predictedExpression shape is {predictedExpression.X.shape} on {experiment_name}.")
    if not expression.X.shape[1] == baseline_observed.X.shape[1]:
        raise ValueError(f"expression and baseline must have the same number of genes on experiment {experiment_name}.")
    if not len(predictedExpression.obs_names) == len(expression.obs_names):
        raise ValueError(f"expression and predictedExpression must have the same size .obs on experiment {experiment_name}.")
    if not all(predictedExpression.obs_names == expression.obs_names):
        raise ValueError(f"expression and predictedExpression must have the same indices on experiment {experiment_name}.")
    baseline_predicted = baseline_predicted.X.mean(axis=0).squeeze()
    baseline_observed = baseline_observed.X.mean(axis=0).squeeze()
    metrics_per_target = evaluate_across_targets(expression, predictedExpression)
    metrics = evaluate_across_perts(
            expression = expression,
            predictedExpression = predictedExpression, 
            baseline_predicted = baseline_predicted, 
            baseline_observed = baseline_observed,
            experiment_name = experiment_name, 
            classifier = classifier, 
            projector = projector,
                do_careful_checks = do_careful_checks, 
            do_parallel=do_parallel
        )
    print("\nMaking some example plots")
    metrics["spearman"] = metrics["spearman"].astype(float)
    hardest = metrics["spearman"].idxmin()
    easiest = metrics["spearman"].idxmax()
    perturbation_plot_path = os.path.join(outputs, "perturbations", str(experiment_name))
    if doPlots:
        for pert in metrics.index:
            is_hardest = hardest==pert
            is_easiest = easiest==pert
            if is_hardest | is_easiest:
                i = expression.obs["perturbation"]==pert
                observed  = safe_squeeze(expression[         i,:].X.mean(axis=0))
                predicted = safe_squeeze(predictedExpression[i,:].X.mean(axis=0))
                os.makedirs(perturbation_plot_path, exist_ok = True)
                diagonal = alt.Chart(
                    pd.DataFrame({
                        "x":[-1, 1],
                        "y":[-1,1 ],
                    })
                ).mark_line(color= 'black').encode(
                    x= 'x',
                    y= 'y',
                )
                scatterplot = alt.Chart(
                    pd.DataFrame({
                        "Observed log fc": observed-baseline_observed, 
                        "Predicted log fc": predicted-baseline_predicted, 
                        "Baseline expression":baseline_observed,
                    })
                ).mark_circle().encode(
                    x="Observed log fc:Q",
                    y="Predicted log fc:Q",
                    color="Baseline expression:Q",
                ).properties(
                    title=pert + " (Spearman rho="+ str(round(metrics.loc[pert,"spearman"], ndigits=2)) +")"
                ) + diagonal
                alt.data_transformers.disable_max_rows()
                pd.DataFrame().to_csv(os.path.join(perturbation_plot_path, f"{pert}.txt"))
                try:
                    scatterplot.save(os.path.join(perturbation_plot_path, f"{pert}.svg"))
                    if is_easiest:
                        scatterplot.save(os.path.join(perturbation_plot_path, f"_easiest({pert}).svg"))
                    if is_hardest:
                        scatterplot.save(os.path.join(perturbation_plot_path, f"_hardest({pert}).svg"))
                except Exception as e:
                    print(f"Altair saver failed with error {repr(e)}")
    metrics["perturbation"] = metrics.index
    return [metrics, metrics_per_target]
    
def assert_perturbation_metadata_match(
        predicted: anndata.AnnData,
        observed: anndata.AnnData, 
        fields_to_check = ["perturbation", "expression_level_after_perturbation"]
    ):
    """Raise an error if the perturbation metadata does not match between observed and predicted anndata."""
    try:
        assert predicted.shape[0]==observed.shape[0]
    except AssertionError:
        print(f"Object shapes: (observed, predicted):", flush = True)
        print((predicted.shape, observed.shape), flush = True)
        raise AssertionError("Predicted and observed anndata are different shapes.")
    predicted.obs_names = observed.obs_names
    for c in ["perturbation", "expression_level_after_perturbation"]:
        if not all(
                predicted.obs[c].astype(str) == observed.obs[c].astype(str)
            ):
            print(predicted.obs[c].head())
            print(observed.obs[c].head())
            raise AssertionError(f"{c} is different between observed and predicted.")
    return