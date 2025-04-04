import os
import numpy as np
import pandas as pd
import scanpy as sc
import argparse
from argparse import Namespace
import datetime
import gc 
import time 
try:
    import memray
    IS_MEMRAY_AVAILABLE = True
except ImportError:
    IS_MEMRAY_AVAILABLE = False
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
parser.add_argument('--do_memory_profiling', dest='do_memory_profiling', action='store_true', help="If this flag is used, use memray for memory profiling.")
parser.add_argument('--networks', type=str, default='../network_collection/networks', help="Location of our network collection on your hard drive")
parser.add_argument('--data', type=str, default='../perturbation_data/perturbations', help="Location of our perturbation data on your hard drive")
parser.add_argument('--tf', type=str, default = "../accessory_data/tf_lists",     help="Location of per-species lists of TFs on your hard drive")
parser.add_argument('--output', type=str, default = "experiments",     help="Folder to save the output in.")
parser.add_argument('--input', type=str, default = "experiments",     help="metadata.json should be in <input>/<experiment_name>/metadata.json.")
parser.add_argument('--verbosity', type=int, default = 1,     help="How much to print out; 0 or 1 or 2.")
parser.add_argument(
    "--amount_to_do",
    choices = ["evaluations", "models", "missing_models"],
    default = "missing_models",
    help="""
    To redo just evaluations using saved predictions, specify "evaluations".
    To redo everything, specify "models". 
    To do predictions whenever they are not yet saved (e.g. pick up an interrupted run), specify "missing_models". 
    To skip certain models (e.g. skip ExtraTrees if low on RAM), manually place 
    an empty results file like 'touch experiments/my_experiment/outputs/results/predictions/3.h5ad'.
    """
)
parser.set_defaults(feature=True)
args = parser.parse_args()
if args.verbosity >= 1:
    print("args to experimenter.py:", flush = True)
    print(args)

# Access our data collections
pereggrn_networks.set_grn_location(
    args.networks
)
pereggrn_perturbations.set_data_path(
    args.data
)
# Default args to this script for interactive use
if args.experiment_name is None:
    args = Namespace(**{
        "experiment_name": "1.8.2_1",
        "amount_to_do": "missing_models",
        "save_trainset_predictions": False,
        "output": "experiments",
        "input": "experiments",
        "tf": "../accessory_data/tf_lists",
        "save_models": False,
        "skip_bad_runs": False, # Makes debug/traceback easier
        "no_parallel": True, # Makes debug/traceback easier
        "do_memory_profiling": False,
        "verbosity": 2,
    })

# Additional bookkeeping
print("Running experiment", flush = True)
outputs = os.path.join(args.output, args.experiment_name, "outputs")
os.makedirs(outputs, exist_ok=True)
metadata = experimenter.validate_metadata(experiment_name=args.experiment_name, input_folder=args.input)
if args.verbosity >= 1:
    print("Starting at " + str(datetime.datetime.now()), flush = True)

try:
    tf_list_path = os.path.join(args.tf, metadata["species"] + ".txt")
    TF_LIST = pd.read_csv(tf_list_path, header = None)[0]
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find the TF list for {metadata['species']} at {tf_list_path}. Please check the species name (in metadata.json) and the accessory data location (the --tf arg to the pereggrn command line tool).")

# Set up the perturbation and network data
perturbed_expression_data, networks, conditions, timeseries_expression_data, screen = experimenter.set_up_data_networks_conditions(
    metadata,
    amount_to_do = args.amount_to_do, 
    outputs = outputs,
)

# Split the data
def get_current_data_split(i, verbose = False, perturbed_expression_data = perturbed_expression_data):
    if conditions.loc[i, "merge_replicates"]:
        perturbed_expression_data = experimenter.averageWithinPerturbation(ad=perturbed_expression_data)
        if timeseries_expression_data is not None:
            raise ValueError("We do not currently support merging of replicates in time-series training data.")
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
os.makedirs(os.path.join( outputs, "predictions"   ),     exist_ok=True) 
os.makedirs(os.path.join( outputs, "predictions_screen"), exist_ok=True) 
os.makedirs(os.path.join( outputs, "fitted_values" ),     exist_ok=True) 
os.makedirs(os.path.join( outputs, "train_resources" ),   exist_ok=True) 
os.makedirs(os.path.join( outputs, "train_memory_requirements" ),    exist_ok=True) 

for i in conditions.index:
    models          = os.path.join( outputs, "models",        str(i) )
    h5ad            = os.path.join( outputs, "predictions",   str(i) + ".h5ad" )
    h5ad_fitted     = os.path.join( outputs, "fitted_values", str(i) + ".h5ad" )
    h5ad_screen     = os.path.join( outputs, "predictions_screen", str(i) + ".h5ad" )
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
                if args.verbosity >= 1:
                    print(f"Fitting model for condition {i} at " + str(datetime.datetime.now()), flush = True)
                    print(conditions.loc[i,:].T)
                start_time = time.time()
                try:
                    os.unlink(train_mem_file)
                except FileNotFoundError:
                    pass
                if args.do_memory_profiling and not IS_MEMRAY_AVAILABLE:
                    print("We could not import memray, so we cannot do memory profiling.")
                if args.do_memory_profiling and IS_MEMRAY_AVAILABLE:
                    with memray.Tracker(train_mem_file, follow_fork = True, file_format = memray.FileFormat.AGGREGATED_ALLOCATIONS): 
                        grn = experimenter.do_one_run(
                            conditions = conditions, 
                            i = i,
                            train_data = perturbed_expression_data_train_i, 
                            test_data  = perturbed_expression_data_heldout_i,
                            networks = networks, 
                            outputs = outputs,
                            metadata = metadata,
                            tfs = TF_LIST,
                            do_parallel=(not args.no_parallel),
                        )
                else: 
                    grn = experimenter.do_one_run(
                        conditions = conditions, 
                        i = i,
                        train_data = perturbed_expression_data_train_i, 
                        test_data  = perturbed_expression_data_heldout_i,
                        networks = networks, 
                        outputs = outputs,
                        metadata = metadata,
                        tfs = TF_LIST,
                        do_parallel=(not args.no_parallel),
                    )
                train_time = time.time() - start_time
                try:
                    peak_ram = subprocess.run(["memray", "summary", train_mem_file], capture_output=True, text=True).stdout
                    peak_ram = peak_ram.split("\n")
                    peak_ram = [p for p in peak_ram if ("B" in p)][0]
                    peak_ram = peak_ram.split("│")[2].strip()
                except:
                    if args.verbosity >= 1:
                        print(f"Memory profiling results are not found or not as expected. If you passed in --do_memory_profiling, you may find the raw memray output in {train_mem_file} and try to parse it yourself, then save it to {train_time_file}.")
                    peak_ram = np.nan
                pd.DataFrame({"walltime (seconds)":train_time, "peak RAM": peak_ram}, index = [i]).to_csv(train_time_file)
            except Exception as e: 
                if args.skip_bad_runs:
                    print(f"Caught exception\n\n{repr(e)}  during training on experiment {i}; skipping.")
                else:
                    raise e
                continue

            if args.save_models:
                if args.verbosity >= 1:
                    print("Saving models...", flush = True)
                grn.save_models( models )
            
            # Make predictions on test and (maybe) train set
            print("Generating predictions...", flush = True)
            # For backwards compatibility, we allow datasets with no timepoint or cell type information
            for c in ["timepoint", "cell_type"]:
                if c not in perturbed_expression_data_heldout_i.obs.columns or c not in perturbed_expression_data_train_i.obs.columns:
                    perturbed_expression_data_heldout_i.obs[c] = 0
                    perturbed_expression_data_train_i.obs[c] = 0            
            experimenter.make_predictions(
                perturbed_expression_data_heldout_i,
                perturbed_expression_data_train_i,
                screen,
                conditions,
                i,
                grn,
                skip_bad_runs = args.skip_bad_runs,
                no_parallel = args.no_parallel,
                save_trainset_predictions = args.save_trainset_predictions, 
                h5ad = h5ad,
                h5ad_fitted = h5ad_fitted,
                h5ad_screen = h5ad_screen,
            )
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
    if args.verbosity >= 1:
        print("Checking sizes: ", flush = True)
    for i in conditions.index:
        if conditions.loc[i, "type_of_split"] != "timeseries":
            if args.verbosity >= 1:
                print(f"- {i}", flush=True)
            perturbed_expression_data_train_i, perturbed_expression_data_heldout_i = get_current_data_split(i)
            evaluator.assert_perturbation_metadata_match(predictions[i], perturbed_expression_data_heldout_i)
            del perturbed_expression_data_train_i
            del perturbed_expression_data_heldout_i
            gc.collect()

    if args.verbosity >= 1:
        print("Evaluating against perturbation transcriptomic data", flush = True)
    evaluationPerPert, evaluationPerTarget = evaluator.evaluateCausalModel(
        get_current_data_split = get_current_data_split, 
        predicted_expression =  predictions,
        is_test_set = True,
        conditions = conditions.copy(), # it corrupts it if you pass by reference. rawr.
        outputs = outputs,
        do_scatterplots = False,
        do_parallel = not args.no_parallel, 
        verbosity=args.verbosity,
        skip_bad_runs=args.skip_bad_runs,
    )
    # pyarrow cannot handle int 
    if args.verbosity >= 1:
        print("Sanitizing evaluation results", flush = True)
    evaluator.convert_to_simple_types(evaluationPerPert, types = [float, str]).to_parquet(   os.path.join(outputs, "evaluationPerPert.parquet"))
    evaluator.convert_to_simple_types(evaluationPerTarget, types = [float, str]).to_parquet( os.path.join(outputs, "evaluationPerTarget.parquet"))
    if fitted_values is not None:
        if args.verbosity >= 1:
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
            verbosity=args.verbosity,
            skip_bad_runs=args.skip_bad_runs,
        )
        os.makedirs(os.path.join(outputs, "trainset_performance"), exist_ok=True)
        evaluationPerPertTrainset.to_parquet(   os.path.join(outputs, "trainset_performance", "evaluationPerPert.parquet"))
        evaluationPerTargetTrainset.to_parquet( os.path.join(outputs, "trainset_performance", "evaluationPerTarget.parquet"))

    if screen is not None:
        predictions_screen = {i:sc.read_h5ad( os.path.join(outputs, "predictions_screen",   str(i) + ".h5ad" ), backed='r' ) for i in conditions.index}
        if args.verbosity >= 1:
            print("Evaluating against single-phenotype screen data", flush = True)
        evaluator.evaluateScreen(
            get_current_data_split,
            predicted_expression=predictions_screen,
            conditions=conditions,
            outputs=outputs,
            screen=screen, 
            skip_bad_runs=args.skip_bad_runs,
        )

else:
    raise ValueError("--amount_to_do must be one of the allowed options (see them on the help page by passing the -h flag).")

print("Experiment done at " + str(datetime.datetime.now()), flush = True)

# Avoid a needless error by defining main.
# This has to do with how python packages typically make command line scripts available.
# A way generally considered better would be to put all the code in main() or parse_args(), then call main() at the end.
# This helps if you want to import this script and do e.g. unit tests on individual functions. 
def main():
    return