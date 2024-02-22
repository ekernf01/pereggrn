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
import altair as alt
import perturbation_benchmarking_package.experimenter as experimenter
from  scipy.stats import chi2_contingency
from scipy.stats import f_oneway

def test_targets_vs_non_targets( predicted, observed, baseline ): 
    targets_positive = np.sign(np.round( predicted - baseline, 2))== 1
    targets_negative = np.sign(np.round( predicted - baseline, 2))==-1
    non_targets      = np.sign(np.round( predicted - baseline, 2))== 0
    fc_observed = observed - baseline
    if any(targets_positive) and any(targets_negative) and any(non_targets):
        return f_oneway(
            fc_observed[targets_positive],
            fc_observed[targets_negative],
            fc_observed[non_targets],
        ).pvalue
    elif any(non_targets) and any(targets_negative):
        return f_oneway(
            fc_observed[targets_negative],
            fc_observed[non_targets],
        ).pvalue    
    elif any(targets_positive) and any(targets_negative):
        return f_oneway(
            fc_observed[targets_positive],
            fc_observed[targets_negative],
        ).pvalue
    elif any(targets_positive) and any(non_targets):
        return f_oneway(
            fc_observed[targets_positive],
            fc_observed[non_targets],
        ).pvalue
    else:
        return np.nan
    

def fc_targets_vs_non_targets( predicted, observed, baseline ): 
    targets_positive = np.sign(np.round( predicted - baseline, 2))== 1
    targets_negative = np.sign(np.round( predicted - baseline, 2))==-1
    non_targets      = np.sign(np.round( predicted - baseline, 2))== 0
    fc_observed = observed - baseline
    if any(targets_positive) and any(targets_negative) and any(non_targets):
        return fc_observed[targets_positive].mean() - fc_observed[targets_negative].mean()
    elif any(non_targets) and any(targets_negative):
        return fc_observed[non_targets].mean() - fc_observed[targets_negative].mean()
    elif any(targets_positive) and any(targets_negative):
        return fc_observed[targets_positive].mean() - fc_observed[targets_negative].mean()
    elif any(targets_positive) and any(non_targets):
        return fc_observed[targets_positive].mean() - fc_observed[non_targets].mean()
    else:
        return np.nan

METRICS = {
    "spearman":                     lambda predicted, observed, baseline: [x for x in spearmanr(observed - baseline, predicted - baseline)][0],
    "mae":                          lambda predicted, observed, baseline: np.abs(observed - predicted).mean(),
    "mse":                          lambda predicted, observed, baseline: np.linalg.norm(observed - predicted)**2,
    "mse_top_20":                   lambda predicted, observed, baseline: mse_top_n(predicted, observed, baseline, n=20),
    "mse_top_100":                  lambda predicted, observed, baseline: mse_top_n(predicted, observed, baseline, n=100),
    "mse_top_200":                  lambda predicted, observed, baseline: mse_top_n(predicted, observed, baseline, n=200),
    "proportion_correct_direction": lambda predicted, observed, baseline: np.mean(np.sign(observed - baseline) == np.sign(predicted - baseline)),
    "pvalue_effect_direction":      lambda predicted, observed, baseline: chi2_contingency(
        observed = pd.crosstab(
            np.sign(np.round( observed - baseline, 2)),
            np.sign(np.round(predicted - baseline, 2))
        )
    ).pvalue,
    "pvalue_targets_vs_non_targets":  test_targets_vs_non_targets,
    "fc_targets_vs_non_targets": fc_targets_vs_non_targets,
}

def mse_top_n(predicted, observed, baseline, n):
    top_n = rank(-np.abs(observed - baseline)) <= n
    return np.linalg.norm(observed[top_n] - predicted[top_n]) ** 2

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
):
    """Compile plots and tables comparing heldout data and predictions for same. 

    Args:
        get_current_data_split: function to retrieve tuple of anndatas (train, test)
        predicted_expression: dict with keys equal to the index in "conditions" and values being anndata objects. 
        is_test_set: True if the predicted_expression is on the test set and False if predicted_expression is on the training data.
        conditions (pd.DataFrame): Metadata for the different combinations used in this experiment. 
        outputs (String): Saves output here.
    """
    evaluationPerPert = {}
    evaluationPerTarget = {}
    evaluations  = []
    for i in predicted_expression.keys():
        perturbed_expression_data_train_i, perturbed_expression_data_heldout_i = get_current_data_split(i)
        evaluations = evaluateOnePrediction(
            expression = perturbed_expression_data_heldout_i if is_test_set else perturbed_expression_data_train_i,
            predictedExpression = predicted_expression[i],
            baseline = perturbed_expression_data_train_i[[bool(b) for b in perturbed_expression_data_train_i.obs["is_control"]], :],
            doPlots=do_scatterplots,
            outputs = outputs,
            experiment_name = i,
            classifier = experimenter.train_classifier(perturbed_expression_data_train_i, target_key = classifier_labels),
            do_parallel=do_parallel
        )
        # Add detail on characteristics of each gene that might make it more predictable
        evaluationPerPert[i],   _ = addGeneMetadata(evaluations[0], genes_considered_as="perturbations", adata=perturbed_expression_data_train_i, adata_test=perturbed_expression_data_heldout_i, path_to_accessory_data=path_to_accessory_data)
        evaluationPerTarget[i], _ = addGeneMetadata(evaluations[1], genes_considered_as="targets"      , adata=perturbed_expression_data_train_i, adata_test=perturbed_expression_data_heldout_i, path_to_accessory_data=path_to_accessory_data)
        evaluationPerPert[i]["index"]   = i
        evaluationPerTarget[i]["index"] = i

    # Concatenate and add some extra info
    evaluationPerPert = postprocessEvaluations(evaluationPerPert, conditions)
    evaluationPerTarget = postprocessEvaluations(evaluationPerTarget, conditions)
    return evaluationPerPert, evaluationPerTarget

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
        baseline: np.matrix, 
        classifier, 
    ) -> pd.DataFrame:
    """Calculate evaluation metrics for one perturbation. 

    Args:
        pert (str): name(s) of perturbed gene(s)
        all_perts (pd.Series): name(s) of perturbed gene(s), one per sample in predictedExpression
        expression (np.matrix): actual expression, log1p-scale. We use a matrix, not an AnnData, for fast parallelism via memory sharing.
        predictedExpression (np.matrix): predicted expression, log1p-scale
        baseline (np.matrix): baseline expression, log1p-scale
        classifier (optional): None or sklearn classifier to judge results by cell type label accuracy

    Returns:
        pd.DataFrame: Evaluation results for each perturbation
    """
    i = all_perts==pert
    predicted = safe_squeeze(predictedExpression[i, :].mean(axis=0))
    observed = safe_squeeze(expression[i, :].mean(axis=0))
    assert observed.shape[0] == expression.shape[1], f"For perturbation {pert}, observed and predicted are different shapes."
    def is_constant(x):
        return np.std(x) < 1e-12
    if any(np.isnan(predicted)) or is_constant(predicted - baseline) or is_constant(observed - baseline):
        return pd.DataFrame({m:np.nan for m in METRICS.keys()}, index = [pert])
    else:
        results = {k:m(predicted, observed, baseline) for k,m in METRICS.items()}
        results["cell_type_correct"] = np.nan
        if classifier is not None:
            class_observed = classifier.predict(np.reshape(observed, (1, -1)))[0]
            class_predicted = classifier.predict(np.reshape(predicted, (1, -1)))[0]
            results["cell_type_correct"] = 1.0 * (class_observed == class_predicted)
        return pd.DataFrame(results, index = [pert])

def evaluate_across_perts(expression: anndata.AnnData, 
                          predictedExpression: anndata.AnnData, 
                          baseline: anndata.AnnData, 
                          experiment_name: str, 
                          classifier, 
                          do_careful_checks: bool=False, 
                          do_parallel: bool = True) -> pd.DataFrame:
    """Evaluate performance for each perturbation.

    Args:
        expression (anndata.AnnData): actual expression, log1p-scale
        predictedExpression (anndata.AnnData): predicted expression, log1p-scale
        baseline (anndata.AnnData): baseline expression, log1p-scale
        experiment_name (str): name of the experiment
        classifier (optional): None or sklearn classifier to judge results by cell type label instead of logfc
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
        if not all(
            expression.obs.loc         [:, ["perturbation", elap]].fillna(0) == 
            predictedExpression.obs.loc[:, ["perturbation", elap]].fillna(0)
        ):
            raise ValueError(f"Expression and predicted expression are different sizes or are differently named in experiment {experiment_name}.")
    with parallel_config(temp_folder='/tmp', backend='threading'):
        results = Parallel(n_jobs=cpu_count())(
            delayed(evaluate_per_pert)(pert, expression.obs["perturbation"], expression.X, predictedExpression.X, baseline, classifier) 
            for pert in perts
        )
    results = pd.concat([r for r in results if type(r) == pd.DataFrame])
    return results

def evaluateOnePrediction(
    expression: anndata.AnnData, 
    predictedExpression: anndata.AnnData, 
    baseline: anndata.AnnData, 
    outputs,
    experiment_name: str,
    doPlots=False, 
    classifier=None, 
    do_careful_checks = True, 
    do_parallel: bool = True):
    '''Compare observed against predicted, for expression, fold-change, or cell type.

            Parameters:
                    expression (AnnData): 
                        the observed expression post-perturbation (log-scale in expression.X). 
                    predictedExpression (AnnData): 
                        the cellOracle prediction (log-scale). Elements of predictedExpression.X may be np.nan for 
                        missing predictions, often one gene missing from all samples or one sample missing for all genes.
                        predictedExpression.obs must contain columns "perturbation" (symbol of targeted gene) 
                        and "expression_level_after_perturbation" (e.g. 0 for knockouts). 
                    baseline (AnnData): 
                        control expression level (log-scale)
                    outputs (str): Folder to save output in
                    classifier (sklearn classifier): 
                        machine learning classifier to assign cell type to predicted expression profiles. 
                        Must have a predict() method capable of taking a value from expression or predictedExpression and returning a single class label. 
                    doPlots (bool): Make a scatterplot showing observed vs predicted, one dot per gene. 
                    do_careful_checks (bool): check gene name and expression level associated with each perturbation.
                        They must match between expression and predictionExpression.
            Returns:
                    Pandas DataFrame with Spearman correlation between predicted and observed 
                    log fold change over control.
    '''
    "log fold change using Spearman correlation and (optionally) cell fate classification."""
    if not expression.X.shape == predictedExpression.X.shape:
        raise ValueError(f"expression shape is {expression.X.shape} and predictedExpression shape is {predictedExpression.X.shape} on {experiment_name}.")
    if not expression.X.shape[1] == baseline.X.shape[1]:
        raise ValueError(f"expression and baseline must have the same number of genes on experiment {experiment_name}.")
    if not len(predictedExpression.obs_names) == len(expression.obs_names):
        raise ValueError(f"expression and predictedExpression must have the same size .obs on experiment {experiment_name}.")
    if not all(predictedExpression.obs_names == expression.obs_names):
        raise ValueError(f"expression and predictedExpression must have the same indices on experiment {experiment_name}.")
    baseline = baseline.X.mean(axis=0).squeeze()
    metrics_per_target = evaluate_across_targets(expression, predictedExpression)
    metrics = evaluate_across_perts(expression, predictedExpression, baseline, experiment_name, classifier, do_careful_checks, do_parallel)
    
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
                        "Observed log fc": observed-baseline, 
                        "Predicted log fc": predicted-baseline, 
                        "Baseline expression":baseline,
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
    return metrics, metrics_per_target
    