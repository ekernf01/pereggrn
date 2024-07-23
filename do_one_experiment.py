import os
import numpy as np
import pandas as pd
import scanpy as sc
import argparse
from argparse import Namespace
import datetime
import gc 
import time 
import memray
import subprocess

# Access our code
import pereggrn.evaluator as evaluator
import pereggrn.experimenter as experimenter
import pereggrn_perturbations
import pereggrn_networks

# User input: name of experiment and whether to fully rerun or just remake plots. 
parser = argparse.ArgumentParser("experimenter")
parser.add_argument("--experiment_name", help="Unique id for the experiment.", type=str)
parser.add_argument("--save_models",     help="If true, save model objects.", default = False, action = "store_true")
parser.add_argument("--save_trainset_predictions", help="If provided, make & save predictions of training data.", default = False, action = "store_true")
parser.add_argument("--no_parallel", help="If provided, don't use loky parallelization.", default = False, action = "store_true")
parser.add_argument('--no_skip_bad_runs', dest='skip_bad_runs', action='store_false', help="Unless this flag is used, keep running when some runs hit errors.")
parser.add_argument('--networks', type=str, default='../network_collection/networks', help="Location of our network collection on your hard drive")
parser.add_argument('--data', type=str, default='../perturbation_data/perturbations', help="Location of our perturbation data on your hard drive")
parser.add_argument('--tf', type=str, default = "../accessory_data/humanTFs.csv",     help="Location of our list of TFs on your hard drive")
parser.add_argument('--output', type=str, default = "experiments",     help="Folder to save the output in.")
parser.add_argument('--input', type=str, default = "experiments",     help="metadata.json should be in <input>/<experiment_name>/metadata.json.")

parser.set_defaults(feature=True)
parser.add_argument(
    "--amount_to_do",
    choices = ["plots", "evaluations", "models", "missing_models"],
    default = "missing_models",
    help="""
    The code makes models, evaluations, and plots, in that order. It saves the models and the evaluations. 
    To do just plots, using saved evaluations and models, specify "plots".
    To do plots and evaluations using saved models, specify "evaluations".
    To do everything, specify "models". 
    If it crashes, specify "missing_models" to keep previous progress. 
    To skip certain models (e.g. skip ExtraTrees if low on RAM), manually place 
    an empty results file like 'touch outputs/results/predictions/3.h5ad'.
    """
)
args = parser.parse_args()
print("args to experimenter.py:", flush = True)
print(args)

# Access our data collections
pereggrn_networks.set_grn_location(
    args.networks
)
pereggrn_perturbations.set_data_path(
    args.data
)
try:
    DEFAULT_HUMAN_TFs = pd.read_csv(args.tf)
    DEFAULT_HUMAN_TFs = DEFAULT_HUMAN_TFs.loc[DEFAULT_HUMAN_TFs["Is TF?"]=="Yes", "HGNC symbol"]
except Exception as e:
    raise(f"TF list given in --tf was not found or was not in the right format. The specific error is: {repr(e)}")

# Default args to this script for interactive use
if args.experiment_name is None:
    args = Namespace(**{
        "experiment_name": "1.8.1_0",
        "amount_to_do": "missing_models",
        "save_trainset_predictions": False,
        "output": "experiments",
        "input": "experiments",
        "save_models": False,
        "skip_bad_runs": False, # Makes debug/traceback easier
        "no_parallel": False, # Makes debug/traceback easier
    })
# Additional bookkeeping
print("Running experiment", flush = True)
outputs = os.path.join(args.output, args.experiment_name, "outputs")
os.makedirs(outputs, exist_ok=True)
metadata = experimenter.validate_metadata(experiment_name=args.experiment_name, input_folder=args.input)
print("Starting at " + str(datetime.datetime.now()), flush = True)

# Set up the perturbation and network data
perturbed_expression_data, networks, conditions, timeseries_expression_data = experimenter.set_up_data_networks_conditions(
    metadata,
    amount_to_do = args.amount_to_do, 
    outputs = outputs,
)

# Split the data
def get_current_data_split(i, verbose = False):
    perts = experimenter.filter_genes( perturbed_expression_data, num_genes = conditions.loc[i, "num_genes"], outputs = outputs)
    if conditions.loc[i, "type_of_split"] == "timeseries":
        return timeseries_expression_data[:, perts.var_names], perts
    return experimenter.splitDataWrapper(
        perts,
        networks = networks, 
        desired_heldout_fraction                 = conditions.loc[i, "desired_heldout_fraction"],  
        type_of_split                            = conditions.loc[i, "type_of_split"],
        data_split_seed                          = conditions.loc[i, "data_split_seed"],
        allowed_regulators_vs_network_regulators = conditions.loc[i, "allowed_regulators_vs_network_regulators"],
        verbose = verbose,
    )

# Begin conditions
os.makedirs(os.path.join( outputs, "predictions"   ),   exist_ok=True) 
os.makedirs(os.path.join( outputs, "fitted_values" ),   exist_ok=True) 
os.makedirs(os.path.join( outputs, "train_resources" ), exist_ok=True) 
os.makedirs(os.path.join( outputs, "train_memory_requirements" ),    exist_ok=True) 

for i in conditions.index:
    models          = os.path.join( outputs, "models",        str(i) )
    h5ad            = os.path.join( outputs, "predictions",   str(i) + ".h5ad" )
    h5ad_fitted     = os.path.join( outputs, "fitted_values", str(i) + ".h5ad" )
    train_time_file = os.path.join( outputs, "train_resources", f"{i}.csv")
    train_mem_file = os.path.join( outputs, "train_memory_requirements", f"{i}.bin")
    if args.amount_to_do in {"models", "missing_models"}:
        perturbed_expression_data_train_i, perturbed_expression_data_heldout_i = get_current_data_split(i, verbose = True)
        gc.collect()
        # Fit models!!
        if \
            (args.amount_to_do in {"models"}) or \
            (args.amount_to_do in {"missing_models"} and not os.path.isfile(h5ad)):
            try:
                os.unlink(h5ad)
            except FileNotFoundError:
                pass
            try:
                print(f"Fitting model for condition {i} at " + str(datetime.datetime.now()), flush = True)
                print(conditions.loc[i,:].T)
                start_time = time.time()
                try:
                    os.unlink(train_mem_file)
                except FileNotFoundError:
                    pass
                with memray.Tracker(train_mem_file, follow_fork = True, file_format = memray.FileFormat.AGGREGATED_ALLOCATIONS): 
                    grn = experimenter.do_one_run(
                        conditions = conditions, 
                        i = i,
                        train_data = perturbed_expression_data_train_i, 
                        test_data  = perturbed_expression_data_heldout_i,
                        networks = networks, 
                        outputs = outputs,
                        metadata = metadata,
                        human_tfs = DEFAULT_HUMAN_TFs,
                        do_parallel=(not args.no_parallel),
                    )
                train_time = time.time() - start_time
                peak_ram = subprocess.run(["memray", "summary", train_mem_file], capture_output=True, text=True).stdout
                try:
                    peak_ram = peak_ram.split("\n")
                    peak_ram = [p for p in peak_ram if ("B" in p)][0]
                    peak_ram = peak_ram.split("│")[2].strip()
                except:
                    print(f"Memory profiling results are not as expected, but you can find the raw memray output in {train_mem_file} and try to parse it yourself, then save it to {train_time_file}.")
                    peak_ram = np.NAN
                pd.DataFrame({"walltime (seconds)":train_time, "peak RAM": peak_ram}, index = [i]).to_csv(train_time_file)
            except Exception as e: 
                if args.skip_bad_runs:
                    print(f"Caught exception {repr(e)} on experiment {i}; skipping.")
                else:
                    raise e
                continue

            if args.save_models:
                print("Saving models...", flush = True)
                grn.save_models( models )
            
            # Make predictions on test and (maybe) train set
            print("Generating predictions...", flush = True)
            # For backwards compatibility, we allow datasets with no timepoint or cell type information
            for c in ["timepoint", "cell_type"]:
                if c not in perturbed_expression_data_heldout_i.obs.columns or c not in perturbed_expression_data_train_i.obs.columns:
                    perturbed_expression_data_heldout_i.obs[c] = 0
                    perturbed_expression_data_train_i.obs[c] = 0

            # Different defaults for timeseries versus non-timeseries benchmarks. 
            # For timeseries, predict each combo of cell type, timepoint, and perturbation ONCE. No dupes. Use the average expression_level_after_perturbation. 
            # For non-timeseries, predict one value per test datapoint. There may be dupes. 
            # Predictions metadata will be in a dataframe or in the .obs of an AnnData; see docs for grn.predict().
            predictions       = None
            predictions_train = None
            all_except_elap = ['timepoint', 'cell_type', 'perturbation', 'is_control', 'perturbation_type']
            if conditions.loc[i, "type_of_split"] == "timeseries":
                predictions_metadata       = perturbed_expression_data_heldout_i.obs[all_except_elap + ["expression_level_after_perturbation"]].groupby(all_except_elap, observed = True).agg({"expression_level_after_perturbation": experimenter.stringy_mean}).reset_index()
                predictions_train_metadata = perturbed_expression_data_train_i.obs[  all_except_elap + ["expression_level_after_perturbation"]].groupby(all_except_elap, observed = True).agg({"expression_level_after_perturbation": experimenter.stringy_mean}).reset_index()
                assert conditions.loc[i, "starting_expression"] == "control", "cannot currently reveal test data when doing time-series benchmarks"
                # "timepoint" in the prediction metadata is the timepoint at which we START the simulation.
                # We need to arrange it so the simulation also ENDS at a timepoint we can evaluate.
                # In cases where the simulation is long-term, that's a hard problem. 
                # Instead of solving lineage tracing on the fly (lolol), we just start from all available cell types.
                # During evaluation, we will select a subset that "makes sense" to compare to the observed post-perturbation data.
                predictions_metadata = pd.merge(
                    predictions_train_metadata[['timepoint', 'cell_type']].drop_duplicates(), 
                    predictions_metadata[['perturbation', 'is_control', 'perturbation_type', "expression_level_after_perturbation"]],
                    how = "cross", 
                )
                # If there are cell types only present in the test set, try reaching them from any training set cell type. 
                train_set_cell_types = set(predictions_train_metadata["cell_type"])
                test_set_cell_types = set(predictions_metadata["cell_type"]) 
                test_only_cell_types = test_set_cell_types - train_set_cell_types
                if len(test_only_cell_types) > 0:
                    print("Cell types in test set but not in training set:", test_only_cell_types)
                    x = predictions_metadata.query("cell_type in @test_only_cell_types").copy()
                    predictions_metadata = predictions_metadata.query("cell_type in @train_set_cell_types")
                    for ct in train_set_cell_types:
                        x["cell_type"] = ct
                        predictions_metadata = pd.concat([predictions_metadata, x])
                assert any(predictions_metadata["is_control"]), "In timeseries experiments, there should be at least one control condition predicted."                
            else:
                if conditions.loc[i, "starting_expression"] == "control":
                    predictions_metadata       = perturbed_expression_data_heldout_i.obs[all_except_elap+["expression_level_after_perturbation"]]
                    predictions_train_metadata = perturbed_expression_data_train_i.obs[  all_except_elap+["expression_level_after_perturbation"]]
                elif conditions.loc[i, "starting_expression"] == "heldout":
                    print("Setting up initial conditions.")
                    predictions       = perturbed_expression_data_heldout_i.copy()
                    predictions_train = perturbed_expression_data_train_i.copy()                
                    predictions_metadata       = None
                    predictions_train_metadata = None
                else:
                    raise ValueError(f"Unexpected value of 'starting_expression' in metadata: { conditions.loc[i, 'starting_expression'] }")
            try:
                print("Running GRN.predict()...")
                predictions   = grn.predict(
                    predictions = predictions,
                    predictions_metadata = predictions_metadata,
                    control_subtype = conditions.loc[i, "control_subtype"], 
                    feature_extraction_requires_raw_data = grn.feature_extraction.startswith("geneformer"),
                    prediction_timescale = [int(i) for i in conditions.loc[i,"prediction_timescale"].split(",")], 
                    do_parallel = not args.no_parallel,
                )
            except Exception as e: 
                if args.skip_bad_runs:
                    print(f"Caught exception {repr(e)} on experiment {i}; skipping.")
                else:
                    raise e
                continue
            # Output shape should usually match the heldout data in shape.
            if conditions.loc[i, "type_of_split"] != "timeseries":
                assert predictions.shape == perturbed_expression_data_heldout_i.shape, f"There should be one prediction for each observation in the test data. Got {predictions.shape[0]}, expected {perturbed_expression_data_heldout_i.shape[0]}."

            # Sometimes AnnData has trouble saving pandas bool columns and sets, and they aren't needed here anyway.
            try:
                del predictions.obs["is_control"] # we can reconstruct this later via experimenter.find_controls().
                del predictions.obs["is_treatment"] 
                predictions.uns["perturbed_and_measured_genes"]     = list(predictions.uns["perturbed_and_measured_genes"])
                predictions.uns["perturbed_but_not_measured_genes"] = list(predictions.uns["perturbed_but_not_measured_genes"])
            except KeyError as e:
                pass
            print("Saving predictions...")
            experimenter.safe_save_adata( predictions, h5ad )
            del predictions
            
            if args.save_trainset_predictions:
                fitted_values = grn.predict(
                    predictions_metadata = predictions_train_metadata,
                    predictions = predictions_train, 
                    feature_extraction_requires_raw_data = grn.feature_extraction.startswith("geneformer"),
                    prediction_timescale = [int(t) for t in conditions.loc[i,"prediction_timescale"].split(",")],
                    do_parallel = not args.no_parallel,
                )
                fitted_values.obs.index = perturbed_expression_data_train_i.obs.index.copy()
                experimenter.safe_save_adata( fitted_values, h5ad_fitted )
            print("... done.", flush = True)
            del grn
            gc.collect()

# Evaluate the results
if args.amount_to_do in {"models", "missing_models", "evaluations"}:
    print("Retrieving saved predictions", flush = True)
    conditions = experimenter.load_successful_conditions(outputs)
    predictions = {i:sc.read_h5ad( os.path.join(outputs, "predictions",   str(i) + ".h5ad" ), backed='r' ) for i in conditions.index}
    try:
        fitted_values = {i:sc.read_h5ad( os.path.join(outputs, "fitted_values", str(i) + ".h5ad" ), backed='r' ) for i in conditions.index}
    except FileNotFoundError:
        fitted_values = None
    # Check sizes before running all evaluations because it helps catch errors sooner.
    print("Checking sizes: ", flush = True)
    for i in conditions.index:
        if conditions.loc[i, "type_of_split"] != "timeseries":
            print(f"- {i}", flush=True)
            perturbed_expression_data_train_i, perturbed_expression_data_heldout_i = get_current_data_split(i)
            evaluator.assert_perturbation_metadata_match(predictions[i], perturbed_expression_data_heldout_i)       
            del perturbed_expression_data_train_i
            del perturbed_expression_data_heldout_i
            gc.collect()
    
    print("(Re)doing evaluations")
    evaluationPerPert, evaluationPerTarget = evaluator.evaluateCausalModel(
        get_current_data_split = get_current_data_split, 
        predicted_expression =  predictions,
        is_test_set = True,
        conditions = conditions,
        outputs = outputs,
        do_scatterplots = False,
        do_parallel = not args.no_parallel, 

    )
    evaluationPerPert.to_parquet(   os.path.join(outputs, "evaluationPerPert.parquet"))
    evaluationPerTarget.to_parquet( os.path.join(outputs, "evaluationPerTarget.parquet"))
    if fitted_values is not None:
        print("(Re)doing evaluations on (training set predictions)")
        evaluationPerPertTrainset, evaluationPerTargetTrainset = evaluator.evaluateCausalModel(
            get_current_data_split = get_current_data_split, 
            predicted_expression =  fitted_values,
            is_test_set = False,
            conditions = conditions,
            outputs = os.path.join(outputs, "trainset_performance"),
            classifier_labels = None, # Default is to look for "louvain" or give up
            do_scatterplots = False,
            do_parallel = not args.no_parallel,
        )
        os.makedirs(os.path.join(outputs, "trainset_performance"), exist_ok=True)
        evaluationPerPertTrainset.to_parquet(   os.path.join(outputs, "trainset_performance", "evaluationPerPert.parquet"))
        evaluationPerTargetTrainset.to_parquet( os.path.join(outputs, "trainset_performance", "evaluationPerTarget.parquet"))
        
# Plot the results
if args.amount_to_do in {"plots", "models", "missing_models", "evaluations"}:
    print("Retrieving saved predictions", flush = True)
    conditions = experimenter.load_successful_conditions(outputs)
    predictions   = {i:sc.read_h5ad( os.path.join(outputs, "predictions",   str(i) + ".h5ad" ) ) for i in conditions.index}
    try:
        fitted_values = {i:sc.read_h5ad( os.path.join(outputs, "fitted_values", str(i) + ".h5ad" ) ) for i in conditions.index}
    except FileNotFoundError:
        fitted_values = None

    print("Retrieving saved evaluations", flush = True)
    evaluationPerPert   = pd.read_parquet(os.path.join(outputs, "evaluationPerPert.parquet"))
    evaluationPerTarget = pd.read_parquet(os.path.join(outputs, "evaluationPerTarget.parquet"))

    print("Plotting main summaries of results.")
    evaluator.makeMainPlots(
        evaluationPerPert, 
        evaluationPerTarget, 
        outputs = outputs, 
        factor_varied = metadata["factor_varied"],
        color_by = metadata["color_by"],
        facet_by=metadata["facet_by"],
    )
    try:
        evaluationPerPertTrainset   = pd.read_parquet(os.path.join(outputs, "trainset_performance", "evaluationPerPert.parquet"))
        evaluationPerTargetTrainset = pd.read_parquet(os.path.join(outputs, "trainset_performance", "evaluationPerTarget.parquet"))
        evaluator.makeMainPlots(
            evaluationPerPertTrainset, 
            evaluationPerTargetTrainset, 
            outputs = os.path.join(outputs, "trainset_performance"), 
            factor_varied = metadata["factor_varied"],
            color_by = metadata["color_by"],
            facet_by = metadata["facet_by"],
        )
    except FileNotFoundError:
        pass
else:
    raise ValueError("--amount_to_do must be one of the allowed options (see them on the help page by passing the -h flag).")

print("Experiment done at " + str(datetime.datetime.now()), flush = True)

# Avoid a useless error by defining main.
# This has to do with how python packages typically make command line scripts available.
# A way generally considered better would be to put all the code in main() or parse_args(), then call main() at the end.
# This helps if you want to import this script and do e.g. unit tests on it. 
def main(some_args):
    return

# def parse_arguments():
#     argument_parse_code
#     return arguments

# if __name__ == '__main__':
#     arguments = parse_arguments()
#     main(*arguments)