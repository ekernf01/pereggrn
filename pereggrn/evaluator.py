"""evaluator.py is a collection of functions for testing predictions about expression fold change.
"""
from joblib import Parallel, delayed, cpu_count
from joblib.parallel import parallel_config
import numpy as np
import pandas as pd
import anndata
from scipy.stats import spearmanr, pearsonr
from scipy.stats import rankdata as rank
import os 
import altair as alt
import pereggrn.experimenter as experimenter
from  scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from typing import Tuple, Dict, List
import gc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest

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
    "pearson":                      lambda predicted, observed, baseline_predicted, baseline_observed: [x for x in pearsonr(observed - baseline_observed,   predicted - baseline_predicted)][0],
    "mae":                          lambda predicted, observed, baseline_predicted, baseline_observed: np.abs               (observed - baseline_observed - (predicted - baseline_predicted)).mean(),
    "mse":                          lambda predicted, observed, baseline_predicted, baseline_observed: np.linalg.norm       (observed - baseline_observed - (predicted - baseline_predicted))**2,
    "mse_top_20":                   lambda predicted, observed, baseline_predicted, baseline_observed: mse_top_n(predicted, observed, baseline_predicted, baseline_observed, n=20),
    "mse_top_100":                  lambda predicted, observed, baseline_predicted, baseline_observed: mse_top_n(predicted, observed, baseline_predicted, baseline_observed, n=100),
    "mse_top_200":                  lambda predicted, observed, baseline_predicted, baseline_observed: mse_top_n(predicted, observed, baseline_predicted, baseline_observed, n=200),
    "overlap_top_20":               lambda predicted, observed, baseline_predicted, baseline_observed: overlap_top_n(predicted, observed, baseline_predicted, baseline_observed, n=20),
    "overlap_top_100":              lambda predicted, observed, baseline_predicted, baseline_observed: overlap_top_n(predicted, observed, baseline_predicted, baseline_observed, n=100),
    "overlap_top_200":              lambda predicted, observed, baseline_predicted, baseline_observed: overlap_top_n(predicted, observed, baseline_predicted, baseline_observed, n=200),
    "pearson_top_20":               lambda predicted, observed, baseline_predicted, baseline_observed: pearson_top_n(predicted, observed, baseline_predicted, baseline_observed, n=20),
    "pearson_top_100":              lambda predicted, observed, baseline_predicted, baseline_observed: pearson_top_n(predicted, observed, baseline_predicted, baseline_observed, n=100),
    "pearson_top_200":              lambda predicted, observed, baseline_predicted, baseline_observed: pearson_top_n(predicted, observed, baseline_predicted, baseline_observed, n=200),
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

def pearson_top_n(predicted, observed, baseline_predicted, baseline_observed, n):
    top_n = rank(-np.abs(observed - baseline_observed)) <= n
    try: 
        return pearsonr((observed - baseline_observed)[top_n], (predicted - baseline_predicted)[top_n])[0]
    except:
        return 0

def overlap_top_n(predicted, observed, baseline_predicted, baseline_observed, n):
    top_n_observed  = rank(-np.abs( observed -  baseline_observed)) <= n
    top_n_predicted = rank(-np.abs(predicted - baseline_predicted)) <= n
    return np.sum(top_n_observed*top_n_predicted)/n

def mse_top_n(predicted, observed, baseline_predicted, baseline_observed, n):
    top_n = rank(-np.abs(observed - baseline_observed)) <= n
    return np.linalg.norm((observed - baseline_observed - (predicted - baseline_predicted))[top_n]) ** 2

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
            how = "right", # This yields missing values. Will deal with that later
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
            how = "right", # This yields missing values. Will deal with that later
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
            how = "right", # This yields missing values. Will deal with that later
            left_on="gene", 
            right_on="gene"
        )
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



def convert_to_simple_types(df: pd.DataFrame, types = [int, float, str]) -> pd.DataFrame:
    for c in df.columns:
        for t in types:
            try:
                oldtype = df[c].dtype
                assert all(df[c].astype(t).astype(oldtype)==df[c])
                df[c] = df[c].astype(t)
                break
            except:
                pass
    return df    

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
    evaluations   = convert_to_simple_types(evaluations).reset_index()
    return evaluations

def evaluateCausalModel(
    get_current_data_split:callable, 
    predicted_expression: dict,
    is_test_set: bool,
    conditions: pd.DataFrame, 
    outputs: str, 
    classifier_labels = None,
    do_scatterplots = False, 
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
    for i in predicted_expression.keys(): #equivalent to: i in conditions.index
        perturbed_expression_data_train_i, perturbed_expression_data_heldout_i = get_current_data_split(i)
        pca20 = PCA(
            n_components = np.min(
                [
                    20, perturbed_expression_data_train_i.shape[1], perturbed_expression_data_train_i.shape[0]
                ]
            )
        )
        try:
            pca20.fit(perturbed_expression_data_train_i.X.toarray())
        except AttributeError: # data not sparse
            pca20.fit(perturbed_expression_data_train_i.X)
        embedding = conditions.loc[i, "visualization_embedding"]
        if np.isnan(embedding):
            embedding = None
            viz_2d = None
        else:
            viz_2d = make_pipeline(KNeighborsRegressor(n_neighbors=10))
            viz_2d.fit(X = perturbed_expression_data_train_i.X, y = perturbed_expression_data_train_i.obsm[embedding][:,k])
        all_test_data = perturbed_expression_data_heldout_i if is_test_set else perturbed_expression_data_train_i # sometimes we predict the training data.
        evaluations = {}
        if "prediction_timescale" not in predicted_expression[i].obs.columns:
            predicted_expression[i].obs["prediction_timescale"] = conditions.loc[i, "prediction_timescale"]
        timescales = predicted_expression[i].obs["prediction_timescale"].unique()
        predicted_expression[i] = predicted_expression[i].to_memory(copy = True)
        predicted_expression[i] = predicted_expression[i][pd.notnull(predicted_expression[i].X.sum(1)), :]
        predicted_expression[i] = experimenter.find_controls(predicted_expression[i])    # the is_control metadata is not saved by the prediction software. Instead, I reconstruct it. because I'm dumb.
        
        # to maintain backwards compatibility, this allows the timepoint and celltype fields to be missing.
        if not "timepoint" in predicted_expression[i].obs.columns: 
            predicted_expression[i].obs["timepoint"] = 0
        if not "cell_type" in predicted_expression[i].obs.columns: 
            predicted_expression[i].obs["cell_type"] = 0
        
        if not "timepoint" in all_test_data.obs.columns: 
            all_test_data.obs["timepoint"] = 0
        if not "cell_type" in all_test_data.obs.columns: 
            all_test_data.obs["cell_type"] = 0
        
        its_values = [True, False] if "timeseries" in conditions["type_of_split"] else [True]
        for is_timescale_strict in its_values:
            print(f"Evaluating condition {i}, with is_timescale_strict {is_timescale_strict}.")
            evaluations[is_timescale_strict] = dict()
            for prediction_timescale in timescales:
                print(f"    Timescale selected: {prediction_timescale}.")
                predicted_expression_it = predicted_expression[i]
                predicted_expression_it = predicted_expression_it[predicted_expression_it.obs["prediction_timescale"]==prediction_timescale, :]
                if conditions.loc[i, "type_of_split"] == "timeseries":
                    assert any(predicted_expression[i].obs["is_control"]), f"No controls found among predictions when evaluating condition {i}."
                    # For timeseries-versus-perturbseq splits, baseline and observed-to-predicted matching are more complicated. See `docs/timeseries_prediction.md` for details.
                    # this returns anndatas in the order OBSERVED, PREDICTED
                    current_heldout, matched_predictions = select_comparable_observed_and_predicted(
                        conditions = conditions,
                        predictions = predicted_expression_it, 
                        perturbed_expression_data_heldout_i = all_test_data, 
                        i = i,
                        # I don't care if this is somewhat redundant with the classifier used below. We need both even if not elegant.
                        classifier = experimenter.train_classifier(perturbed_expression_data_train_i, target_key = "cell_type"), 
                        is_timescale_strict=is_timescale_strict,
                    )
                    # The sensible baseline differs between predicted and test data. 
                    # For the test data, it should be a **test-set** control sample from the same timepoint and cell type. 
                    # For the predictions, it should be a **prediction under no perturbations** from the same timepoint and cell type. 
                    # Because the upstream code selects perturbations to predict from the test set, the names of the controls should match the heldout data.
                    assert any(current_heldout.obs["is_control"]), f"No controls found among heldout data when evaluating condition {i}, timescale {prediction_timescale}."
                    assert any(matched_predictions.obs["is_control"]), f"No controls found among predictions when evaluating condition {i}, timescale {prediction_timescale}."
                    baseline_observed = current_heldout.copy()[current_heldout.obs["is_control"], :]
                    baseline_predicted = matched_predictions[ matched_predictions.obs["is_control"], : ].copy()
                else:
                    current_heldout = all_test_data
                    matched_predictions = predicted_expression_it
                    # For train-test splits of a single perturbset, the controls are all in the training data. 
                    # The same baseline can be used for the training and test data, and it needs to be extracted from the training data. 
                    assert any(perturbed_expression_data_train_i.obs["is_control"]), "No controls found."
                    baseline_observed  = perturbed_expression_data_train_i[[bool(b) for b in perturbed_expression_data_train_i.obs["is_control"]], :]
                    baseline_predicted = baseline_observed.copy()

                classifier_labels = "cell_type" if (conditions.loc[i, "type_of_split"]=="timeseries") else None # If you pass None, it will look for "louvain" or give up.
                assert "timepoint" in current_heldout.obs.columns
                evaluations[is_timescale_strict][prediction_timescale] = evaluateOnePrediction(
                    expression = current_heldout,
                    predictedExpression = matched_predictions,
                    baseline_observed = baseline_observed,
                    baseline_predicted = baseline_predicted,
                    doPlots=do_scatterplots,
                    outputs = outputs,
                    experiment_name = i,
                    viz_2d = viz_2d,
                    classifier = experimenter.train_classifier(perturbed_expression_data_train_i, target_key = classifier_labels),
                    pca20 = pca20,
                    do_parallel=do_parallel,
                    is_timeseries = (conditions.loc[i, "type_of_split"] == "timeseries"),
                    train = perturbed_expression_data_train_i,
                )
                # Add detail on characteristics of each gene that might make it more predictable
                evaluations[is_timescale_strict][prediction_timescale][0], _ = addGeneMetadata(
                    evaluations[is_timescale_strict][prediction_timescale][0],
                    genes_considered_as="perturbations",
                    adata=perturbed_expression_data_train_i,
                    adata_test=current_heldout, 
                    path_to_accessory_data=path_to_accessory_data
                )
                evaluations[is_timescale_strict][prediction_timescale][1], _ = addGeneMetadata(
                    evaluations[is_timescale_strict][prediction_timescale][1],
                    genes_considered_as="targets",
                    adata=perturbed_expression_data_train_i,
                    adata_test=current_heldout,
                    path_to_accessory_data=path_to_accessory_data
                )
                evaluations[is_timescale_strict][prediction_timescale][0]["is_timescale_strict"] = is_timescale_strict
                evaluations[is_timescale_strict][prediction_timescale][1]["is_timescale_strict"] = is_timescale_strict
                evaluations[is_timescale_strict][prediction_timescale][0]["index"] = i
                evaluations[is_timescale_strict][prediction_timescale][1]["index"] = i
                evaluations[is_timescale_strict][prediction_timescale][0]["prediction_timescale"] = prediction_timescale
                evaluations[is_timescale_strict][prediction_timescale][1]["prediction_timescale"] = prediction_timescale
        print(f"Finished evaluating condition {i}. Concatenating outputs.")
        evaluationPerPert  [i] = pd.concat([evaluations[is_timescale_strict][t][0] for t in timescales for is_timescale_strict in its_values])
        evaluationPerTarget[i] = pd.concat([evaluations[is_timescale_strict][t][1] for t in timescales for is_timescale_strict in its_values])
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
    is_timescale_strict: bool
) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """Select a set of predictions that are comparable to the test data, and aggregate the test data within each combination of
    perturbed gene, timepoint, and cell_type. See docs/timeseries_prediction.md for details.
    This function should NOT be run unless type_of_split is "timeseries".

    Args:
        conditions (pd.DataFrame): all experimental conditions
        predictions (anndata.AnnData): predicted expression
        perturbed_expression_data_heldout_i (anndata.AnnData): the heldout data
        i (int): which condition you are currently preparing to evaluate
        classifier: for methods like PRESCIENT where the simulation can last a long time, we may need to update the "cell_type" labels to something different from the initial state. 
            This should be an sklearn classifier with a .predict() method. It should accept all genes as features.
        is_timescale_strict: if true, make sure the takedown timepoint is equal between the observed and predicted data. If false, select 
            the largest effect and compare the observed data against that. This accommodates a block in differentiation, e.g. SOX17 KD in endoderm.
    Returns:
        Tuple[anndata.AnnData, anndata.AnnData]: the observed test data and the predictions, with a one-to-one match between them.
    """
    # For timeseries datasets, we just return a single prediction for each timepoint and cell_type, not one for each test set cell.
    # So let's average the test data too.
    test_data = experimenter.averageWithinPerturbation(perturbed_expression_data_heldout_i, ["timepoint", "cell_type"])
    # Now we will select predictions for each combo of timepoint, cell_type, and perturbation in the heldout data.
    predictions.obs["takedown_timepoint"] = predictions.obs["timepoint"]
    if conditions.loc[i, "does_simulation_progress"]:
        predictions.obs.loc[:, "takedown_timepoint"] += predictions.obs.loc[:, "prediction_timescale"]
        predictions.obs.loc[:, "cell_type"] = classifier.predict(predictions.X)

    test_data.obs["observed_index"] = test_data.obs.index.copy()
    predictions.obs["predicted_index"] = predictions.obs.index.copy()
    
    # upcoming merge is (john mulaney voice) sensitive about data types
    test_data.obs["perturbation"] = test_data.obs["perturbation"].astype("str")
    predictions.obs["perturbation"] = predictions.obs["perturbation"].astype("str")
    test_data.obs["cell_type"] = test_data.obs["cell_type"].astype("str")
    predictions.obs["cell_type"] = predictions.obs["cell_type"].astype("str")
    test_data.obs["timepoint"] = test_data.obs["timepoint"].astype("Int64")
    predictions.obs["takedown_timepoint"] = predictions.obs["takedown_timepoint"].astype(float).round(0).astype("Int64")
    # Match the timepoint to the observed data ... or not
    if is_timescale_strict:
        matched_predictions = pd.merge(
            test_data.obs[  [         'timepoint',          "cell_type", 'perturbation', "expression_level_after_perturbation", "observed_index"]], 
            predictions.obs[['takedown_timepoint',          "cell_type", 'perturbation', "expression_level_after_perturbation", "predicted_index"]], 
            left_on =['timepoint',          "cell_type", 'perturbation', "expression_level_after_perturbation"],
            right_on=['takedown_timepoint', "cell_type", 'perturbation', "expression_level_after_perturbation"],
            how='inner', 
        )
    else:
        # Select the largest effects from the predictions. 
        predictions.obs["effect_size"] = 0
        for i in predictions.obs.index:
            t,c = predictions.obs.loc[i, ['takedown_timepoint', "cell_type"]]
            matched_control = predictions.obs.query("takedown_timepoint==@t & cell_type==@c & is_control").index
            predictions.obs.loc[i, "effect_size"] = np.linalg.norm( predictions[i, :].X - predictions[matched_control, :].X )
        largest_effects = predictions[predictions.obs.groupby(['perturbation'])['effect_size'].idxmax(), :]
        matched_predictions = pd.merge(
            test_data.obs      [["cell_type", 'perturbation', "expression_level_after_perturbation", "observed_index"]], 
            largest_effects.obs[["cell_type", 'perturbation', "expression_level_after_perturbation", "predicted_index"]], 
            left_on =['perturbation', "expression_level_after_perturbation"],
            right_on=['perturbation', "expression_level_after_perturbation"],
            how='inner', 
        )

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
        group: str,
        predictions_metadata: pd.Series,
        expression: np.matrix, 
        predictedExpression: np.matrix,
        baseline_predicted: np.matrix, 
        baseline_observed: np.matrix, 
        classifier = None, 
        pca20 = None,
        viz_2d = None,
        progenitor: np.matrix = None,
        baseline_training: np.matrix = None,
    ) -> pd.DataFrame:
    """Calculate evaluation metrics for one perturbation. 

    Args:
        group (str): element of predictions_metadata["group"] indicating a collection of predictions that will be evaluated jointly.
            This helps us with metrics that would come out differently if computed on average log FC versus per cell log FC.
            For instance, selecting the top 20 differentially expressed genes is much cleaner when the average is considered, instead of using a different 20 genes for each observed cell. 
        predictions_metadata (pd.DataFrame): metadata for predictions -- perturbation, elap, cell_type, timepoint
        expression (np.matrix): actual expression, log1p-scale. We use a matrix, not an AnnData, for fast parallelism via memory sharing.
        predictedExpression (np.matrix): predicted expression, log1p-scale
        baseline (np.matrix): baseline expression, log1p-scale
        classifier (optional): None or sklearn classifier to judge results by cell type label accuracy
        pca20 (optional): None or sklearn PCA object to project expression into a lower-dimensional space
        viz_2d: fitted model that predicts 2d embedding coordinates from log-scale expression. Trained on the training data.
        baseline_training (np.matrix, optional): average expression of training cells with the same cell type as these predictions and observations. log1p-scale
        progenitor (np.matrix, optional): average expression of progenitors of cells that contributed to baseline_training, log1p-scale

    Returns:
        pd.DataFrame: Evaluation results for each perturbation
    """
    i = predictions_metadata["group"]==group
    predicted = safe_squeeze(predictedExpression[i, :].mean(axis=0))
    observed  = safe_squeeze(         expression[i, :].mean(axis=0))
    assert observed.shape[0] == expression.shape[1], f"For group {group}, observed and predicted are different shapes."
    results = {k:m(predicted, observed, baseline_predicted, baseline_observed) for k,m in METRICS.items()}
    for k in predictions_metadata.columns:
        results[k] = predictions_metadata.loc[i, k].unique()[0]        
    results["num_observations_in_group"] = i.sum()
    results["cell_type_correct"] = np.nan
    if classifier is not None:
        class_observed  = classifier.predict(np.reshape(observed, (1, -1)))[0]
        class_predicted = classifier.predict(np.reshape(predicted, (1, -1)))[0]
        results["cell_type_correct"] = 1.0 * (class_observed == class_predicted)
    results["distance_in_pca"] = np.nan
    if pca20 is not None:
        results["distance_in_pca"] = np.linalg.norm(pca20.transform(observed.reshape(1, -1)) - pca20.transform(predicted.reshape(1, -1)))**2
    if viz_2d is not None:
        viz_embedding_predicted  = viz_2d.transform(predicted - baseline_predicted + baseline_training)
        viz_embedding_progenitor = viz_2d.transform(progenitor[group, :])
        viz_embedding_test       = viz_2d.transform(expression - baseline_observed + baseline_training)
        viz_embedding_train      = viz_2d.transform(baseline_training[group, :])
        results["viz1_predicted"]  = viz_embedding_predicted[0]
        results["viz2_predicted"]  = viz_embedding_predicted[1]
        results["viz1_progenitor"] = viz_embedding_progenitor[0]
        results["viz2_progenitor"] = viz_embedding_progenitor[1]
        results["viz1_test"]       = viz_embedding_test[0]
        results["viz2_test"]       = viz_embedding_test[1]
        results["viz1_train"]      = viz_embedding_train[0]
        results["viz2_train"]      = viz_embedding_train[1]
        results["delay_score"]     = viz_embedding_progenitor.T.dot(viz_embedding_predicted)
    return pd.DataFrame(results, index = [group])

def evaluate_across_perts(
    expression: anndata.AnnData, 
    predictedExpression: anndata.AnnData, 
    baseline_predicted: anndata.AnnData, 
    baseline_observed: anndata.AnnData, 
    train: anndata.AnnData,
    experiment_name: str, 
    classifier = None, 
    pca20 = None,
    do_careful_checks: bool=False, 
    do_parallel: bool = True, 
    viz_2d = None,
    baseline_training = None,
    progenitor = None,
    group_by = ['timepoint', 'cell_type', 'perturbation'],
) -> pd.DataFrame:
    """Evaluate performance for each perturbation.

    Args:
        expression (anndata.AnnData): actual expression, log1p-scale
        predictedExpression (anndata.AnnData): predicted expression, log1p-scale
        baseline_predicted, baseline_observed (anndata.AnnData): baseline expression, log1p-scale
        train (anndata.AnnData): training data
        experiment_name (str): name of the experiment
        classifier (optional): None or sklearn classifier to judge results by cell type label instead of logfc
        pca20 (optional): None or sklearn PCA object to project expression into a lower-dimensional space
        do_careful_checks (bool, optional): ensure that perturbation and dose match between observed
            and predicted expression. Defaults to False.
        do_parallel (bool, optional): if True, use joblib to parallelize the evaluation across perturbations. Defaults to True.
        viz_2d: supervised classifier that predicts 2d embeddings from log-scale expression, trained on the training data 
        baseline_training (np.matrix, optional): average expression of training cells with the same cell type as these predictions and observations. log1p-scale
        progenitor (np.matrix, optional): average expression of progenitors of cells that contributed to baseline_training, log1p-scale
        group_by (list, optional): If many observations have the same values of all these attributes, they will be summarized in one single row of the evaluations.
    Returns:
        pd.DataFrame: _description_
    """
    assert "timepoint" in expression.obs.columns
    predictedExpression.obs["group"] = (1-expression.obs[group_by].duplicated()).cumsum()
    print(f"Evaluating {predictedExpression.obs['group'].max()} groups based on {group_by}.", flush=True)
    predictedExpression = predictedExpression.to_memory()

    # You can't be too careful these days.
    if do_careful_checks:
        elap = "expression_level_after_perturbation"
        predictedExpression.obs[elap] = pd.to_numeric(predictedExpression.obs[elap], errors='coerce')
        expression.obs[elap] = pd.to_numeric(expression.obs[elap], errors='coerce')
        expression.obs[         "perturbation"] = expression.obs[         "perturbation"].astype(str)
        predictedExpression.obs["perturbation"] = predictedExpression.obs["perturbation"].astype(str)
        for c in [ "perturbation", elap ]:
            if not all(
                expression.obs.loc         [:, c].fillna(0) == 
                predictedExpression.obs.loc[:, c].fillna(0)
            ):
                if expression.obs[c].dtype==str or not all(
                    expression.obs.loc         [:, c].fillna(0) - 
                    predictedExpression.obs.loc[:, c].fillna(0) <
                    1e-4
                ):
                    mismatches = expression.obs.loc         [:, c].fillna(0) != predictedExpression.obs.loc[:, c].fillna(0)
                    print("Observed example mismatches:")
                    print(expression.obs.loc         [mismatches, c].fillna(0).head())
                    print("Predicted example mismatches:")
                    print(predictedExpression.obs.loc[mismatches, c].fillna(0).head())
                    raise ValueError(f"Expression and predicted expression have mismatched '{c}' metadata in experiment {experiment_name}. Check stdout for summary statistics.")
    
    # Prepare additional data needed for visualization
    if viz_2d is not None:
        baseline_training = np.zeros((predictedExpression.obs["group"].nunique(), predictedExpression.shape[1]))
        progenitor = baseline_training.copy()
        for group in predictedExpression.obs["group"].unique():
            group_metadata = predictedExpression[predictedExpression.obs["group"]==group, :]
            celltype = group_metadata.obs["cell_type"].unique()[0]
            timepoint = group_metadata.obs["timepoint"].unique()[0]
            train_cells = train.obs["cell_type"]==celltype & train.obs["timepoint"]==timepoint
            progenitor[group, :] = train[train.obs[train_cells, "matched_control"], :].X.mean(0)
            baseline_training[group, :] = train[train_cells, :].X.mean(0)
    else:
        viz_2d = None
        progenitor = None
        baseline_training = None
    
    # Evals via joblib by default. it go fast.
    if do_parallel:
        with parallel_config(temp_folder='/tmp', backend='threading', verbose = 1):
            results = Parallel(n_jobs=cpu_count())(
                delayed(evaluate_per_pert)(
                    group,
                    predictedExpression.obs, 
                    expression.X, 
                    predictedExpression.X, 
                    baseline_predicted, 
                    baseline_observed, 
                    classifier, 
                    pca20, 
                    viz_2d, 
                    progenitor,
                    baseline_training, 
                ) 
                for group in predictedExpression.obs["group"].unique()
            )
    else:
        results = [
            evaluate_per_pert(
                group,                     
                predictedExpression.obs,
                expression.X, 
                predictedExpression.X, 
                baseline_predicted, 
                baseline_observed, 
                classifier, 
                pca20,
                viz_2d,
                progenitor,
                baseline_training,
            ) 
            for group in predictedExpression.obs["group"].unique()
        ]
    print("Finished evaluating all groups. Concatenating.", flush=True)
    results = pd.concat([r for r in results if type(r) == pd.DataFrame])
    return results

def evaluateOnePrediction(
    expression: anndata.AnnData, 
    predictedExpression: anndata.AnnData, 
    baseline_predicted: anndata.AnnData, 
    baseline_observed: anndata.AnnData, 
    outputs,
    experiment_name: str,
    doPlots = False, 
    classifier=None, 
    pca20=None,
    viz_2d = None,
    do_careful_checks = True, 
    do_parallel: bool = True, 
    is_timeseries: bool = False,
    train = anndata.AnnData,
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
                    pca20 (sklearn PCA):
                        PCA or other sklearn dimension reduction object to project expression into a lower-dimensional space.
                    viz_2d: fitted model that predicts 2d embedding coordinates from log-scale expression. Trained on the training data.
                    doPlots (bool): Make a scatterplot showing observed vs predicted, one dot per gene. 
                    do_careful_checks (bool): check gene name and expression level associated with each perturbation.
                        They must match between expression and predictionExpression.
                    do_parallel (bool): use joblib to parallelize the evaluation across perturbations.
                    is_timeseries (bool): for timeseries data we expect a different shape for observed and predicted. The default behavior is to compare
                        predictions to test data within each cell type and timepoint, averaging together all test samples. Also to evaluate predictions 
                        after different numbers of time-steps separately, even if multiple time-steps are returned inside the same AnnData object. 
                    train (AnnData): training data, used to calculate the baseline expression level for each cell type and timepoint.

            Returns:
                    Pandas DataFrame with Spearman correlation between predicted and observed 
                    log fold change over control.
    '''
    "log fold change using Spearman correlation and (optionally) cell fate classification."""
    assert "timepoint" in expression.obs.columns
    if not expression.X.shape == predictedExpression.X.shape:
        raise ValueError(f"expression shape is {expression.X.shape} and predictedExpression shape is {predictedExpression.X.shape} on {experiment_name}.")
    if not expression.X.shape[1] == baseline_observed.X.shape[1]:
        raise ValueError(f"expression and baseline must have the same number of genes on experiment {experiment_name}.")
    if not predictedExpression.X.shape[1] == baseline_predicted.X.shape[1]:
        raise ValueError(f"predictedExpression and baseline must have the same number of genes on experiment {experiment_name}.")
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
            pca20 = pca20,
            do_careful_checks = do_careful_checks, 
            do_parallel = do_parallel, 
            viz_2d = viz_2d,
            baseline_training = None,
            progenitor = None,
            train = train,
        )
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
        # 1 == 1.0 but '1' != '1.0'
        if c=="expression_level_after_perturbation":
            try:
                predicted.obs[c] = predicted.obs[c].astype(float)
                observed.obs[c]  = observed.obs[c].astype(float)
                predicted.obs[c] = predicted.obs[c].round(4)
                observed.obs[c]  = observed.obs[c].round(4)
            except ValueError:
                pass
        if not all(
                predicted.obs[c].astype(str) == observed.obs[c].astype(str)
            ):
            mismatches = predicted.obs[c].astype(str) != observed.obs[c].astype(str)
            print(predicted.obs.loc[mismatches,c].head())
            print(observed.obs.loc[mismatches, c].head())
            raise AssertionError(f"{c} is different between observed and predicted.")
    return

