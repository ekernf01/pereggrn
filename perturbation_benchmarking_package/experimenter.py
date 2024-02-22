import os
import gc
import json
import yaml
import gc
import pandas as pd
import numpy as np
import scipy
import anndata
import scanpy as sc
from itertools import product
import perturbation_benchmarking_package.evaluator as evaluator
import ggrn.api as ggrn
import load_networks
import load_perturbations
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier

def get_required_keys():
    """Get all metadata keys that are required by downstream code. Some have 
    default values so they are not necessarily required to be user-specified. 

    Returns:
        tuple: the keys
    """
    return (
        # Experiment info
        "unique_id",
        "nickname",
        "readme",
        "question",
        "is_active",
        "factor_varied",    
        "color_by",
        "facet_by",
        # Data and preprocessing
        "network_datasets",
        "allowed_regulators_vs_network_regulators",
        "perturbation_dataset",
        "merge_replicates",
        "desired_heldout_fraction",
        "type_of_split",
        # Modeling decisions
        "pruning_parameter", 
        "pruning_strategy",
        "network_prior",
        "regression_method",
        "feature_extraction",
        "low_dimensional_structure",
        "low_dimensional_training",
        "matching_method",
        "prediction_timescale",
    )

def get_optional_keys():
    """Get all metadata keys that are optional in downstream code.
    """
    return (
        "refers_to" ,
        "eligible_regulators",
        "predict_self" ,
        "num_genes",
        "baseline_condition",
        "data_split_seed",
        "starting_expression",
        "control_subtype",
        "expand",
        "kwargs_to_expand",
        "kwargs"
    )

def get_default_metadata():
    """Get default values wherever available for experimental parameters.

    Returns:
        dict: metadata parameter defaults
    """
    return {
        "allowed_regulators_vs_network_regulators": "all",
        "pruning_parameter": None, 
        "pruning_strategy": "none",
        "network_prior": "ignore",
        "desired_heldout_fraction": 0.5,
        "type_of_split": "interventional",
        "regression_method": "RidgeCV",
        "starting_expression": "control",
        "feature_extraction": None,
        "control_subtype": None,
        "kwargs": dict(),
        "kwargs_to_expand": [],
        "expand": "grid",
        "data_split_seed": 0,
        "baseline_condition": 0,
        "num_genes": 10000,
        "predict_self": False,
        'eligible_regulators': "all",
        "merge_replicates": False,
        "network_datasets":{"dense":{}},
        "low_dimensional_structure": "none",
        "low_dimensional_training": "svd",
        "matching_method": "steady_state",
        "prediction_timescale": 1
    }

def validate_metadata(
    experiment_name: str = None,
    metadata: dict = None,
) -> OrderedDict:
    """Make sure the user-provided metadata is OK, and fill in missing info with defaults.

    Args:
        experiment_name (str): name of an Experiment (a folder in the experiments folder)
        metadata (dict): Experiment metadata. 

    Returns:
        OrderedDict: Experiment metadata
    """
    assert experiment_name is None or metadata is None, "You must provide experiment_name or metadata, but not both."
    if experiment_name is not None:
        with open(os.path.join("experiments", experiment_name, "metadata.json")) as f:
            metadata = json.load(f, object_pairs_hook=OrderedDict)
        print("\n\nRaw metadata for experiment " + experiment_name + ":\n")
        print(yaml.dump(metadata))
        assert experiment_name == metadata["unique_id"], "Experiment is labeled right"

    metadata = OrderedDict(metadata)
    if ("is_active" in metadata.keys()) and (not metadata["is_active"]):
        raise ValueError("This experiment is marked as inactive. If you really want to run it, edit its metadata.json.")

    # If metadata refers to another experiment, go find missing metadata there.
    if "refers_to" in metadata.keys():
        with open(os.path.join("experiments", metadata["refers_to"], "metadata.json")) as f:
            other_metadata = json.load(f)
            try:
                assert other_metadata["is_active"], "Referring to an inactive experiment is not allowed."
            except KeyError:
                pass
        for key in other_metadata.keys():
            if key not in metadata.keys():
                metadata[key] = other_metadata[key]
    else:
        metadata["refers_to"] = None

    # Set defaults (None often defers to downstream code)
    defaults = get_default_metadata()
    for k in defaults:
        if not k in metadata:
            metadata[k] = defaults[k]

    # Add some default network handling behavior to reduce metadata boilerplate
    for netName in metadata["network_datasets"].keys():
        if not "subnets" in metadata["network_datasets"][netName].keys():
            metadata["network_datasets"][netName]["subnets"] = []
        if not "do_aggregate_subnets" in metadata["network_datasets"][netName].keys():
            metadata["network_datasets"][netName]["do_aggregate_subnets"] = False
    
    # Check all keys
    required_keys = get_required_keys()
    allowed_keys = required_keys + get_optional_keys()
    missing = [k for k in required_keys if k not in metadata.keys()]
    extra = [k for k in metadata.keys() if k not in allowed_keys]
    assert len(missing)==0, f"Metadata is missing some required keys: {' '.join(missing)}"
    assert len(extra)==0,   f"Metadata has some unwanted keys: {' '.join(extra)}"
    
    # Check a few of the values
    for k in metadata["kwargs_to_expand"]:
        assert k not in metadata, f"Key {k} names both an expandable kwarg and an Experiment metadata key. Sorry, but this is not allowed. See get_default_metadata() and get_required_keys() for names of keys reserved for the benchmarking code."
    assert metadata["perturbation_dataset"] in set(load_perturbations.load_perturbation_metadata().query("is_ready=='yes'")["name"]), "Cannot find perturbation data under that name. Try load_perturbations.load_perturbation_metadata()."
    for netName in metadata["network_datasets"].keys():
        assert netName in set(load_networks.load_grn_metadata()["name"]).union({"dense", "empty"}) or "random" in netName, "Networks exist as named"
        assert "subnets" in metadata["network_datasets"][netName].keys(), "Optional metadata fields filled correctly"
        assert "do_aggregate_subnets" in metadata["network_datasets"][netName].keys(), "Optional metadata fields filled correctly"

    print("\nFully parsed metadata:\n")
    print(yaml.dump(metadata))

    return metadata

def lay_out_runs(
  networks: dict, 
  metadata: dict,
) -> pd.DataFrame:
    """Lay out the specific training runs or conditions included in this experiment.

    Args:
    networks (dict): dict with string keys and LightNetwork values
    outputs (str): folder name to save results in
    metadata (dict): metadata for this Experiment, from metadata.json. See this repo's global README.

    Returns:
        pd.DataFrame: metadata on the different conditions in this experiment

    """
    metadata = metadata.copy() # We're gonna mangle it. :)
    
    # ===== Remove items that don't want to be cartesian-producted =====
    # This is too bulky to want in the csv
    del metadata["readme"]
    
    # Again, too bulky.
    # See experimenter.get_networks() to see how this entry of the metadata.json is parsed.
    # For conditions.csv, we just record the network names.
    metadata["network_datasets"] = list(networks.keys())
    
    # Baseline condition will never be an independently varying factor. 
    baseline_condition = metadata["baseline_condition"]
    try:
        baseline_condition = baseline_condition.copy()
    except AttributeError:
        pass
    del metadata["baseline_condition"]

    # kwargs is a dict containing arbitrary kwargs for backend code (e.g. batch size for DCD-FG). 
    # We allow these to be expanded if the user says so.
    kwargs = metadata["kwargs"].copy()
    kwargs_to_expand = metadata["kwargs_to_expand"].copy()
    del metadata["kwargs"]
    del metadata["kwargs_to_expand"]
    for k in kwargs_to_expand:
        metadata[k] = kwargs[k]
    
    # Wrap each singleton in a list. Otherwise product() and len() will misbehave on strings.
    for k in metadata.keys():
        if type(metadata[k]) != list:
            metadata[k] = [metadata[k]]

    # ==== Done preparing for expansion ====
    if metadata["expand"][0]=="grid":
        del metadata["expand"]
        # Make all combos 
        conditions =  pd.DataFrame(
            [row for row in product(*metadata.values())], 
            columns=metadata.keys()
        )
    elif metadata["expand"][0]=="ladder":
        del metadata["expand"]
        all_lengths = {k:len(v) for k,v in metadata.items()}
        num_conditions = int(np.max(list(all_lengths.values())))
        assert all( [l in (1, num_conditions) for l in all_lengths.values()] ), f"When 'expand' is 'ladder', length must be 1 or the max for all metadata entries. All lengths: {all_lengths}"
        metadata = {k:(v*num_conditions if all_lengths[k]==1 else v) 
                    for k,v in metadata.items()}
        conditions =  pd.DataFrame(metadata)
    else:
        raise ValueError(f"'expand' must be 'grid' or 'ladder'; got {metadata['expand']}")
    
    # Downstream of this, the dense network is stored as an empty network instead of a list of all edges (to save space).
    # Recommended usage is to set network_prior="ignore"; otherwise the empty network will be taken literally. 
    for i in conditions.index:
        conditions.loc[i, "network_prior"] = \
        "ignore" if conditions.loc[i, "network_datasets"] == "dense" else conditions.loc[i, "network_prior"]

    conditions.index.name = "condition"
    conditions["baseline_condition"] = baseline_condition
    return conditions
  
def do_one_run(
    conditions: pd.DataFrame, 
    i: int, 
    train_data: anndata.AnnData, 
    test_data: anndata.AnnData, 
    networks: dict, 
    outputs: str,
    metadata: dict, 
    human_tfs: list,
    do_parallel: bool = True
    ) -> anndata.AnnData:
    """Do one run (fit a GRN model and make predictions) as part of this experiment.

    Args:
        conditions (pd.DataFrame): Output of lay_out_runs
        i (int): A value from the conditions.index
        human_tfs: A list of human TF's. You can pass in None if you never plan to restrict eligible regulators to just TF's.
        Other args: see help(lay_out_runs)

    Returns:
        anndata.AnnData: Predicted expression
    """
    if conditions.loc[i,'eligible_regulators'] == "human_tfs":
        if human_tfs is None:
            raise ValueError("If you want to restrict to only TF's as regulators, provide a list to human_tfs.")
        eligible_regulators = human_tfs 
    elif conditions.loc[i,'eligible_regulators'] == "all":
        eligible_regulators = None # This triggers the default of using all genes
    elif conditions.loc[i,'eligible_regulators'] == "perturbed_genes":
        eligible_regulators = set.union(
            set(train_data.uns["perturbed_and_measured_genes"]),
            set( test_data.uns["perturbed_and_measured_genes"])
        )
    else:
        raise ValueError("'eligible_regulators' must be 'human_tfs' or 'perturbed_genes' or 'all'")
    train_data.obs["is_control"] = train_data.obs["is_control"].astype(bool)
    print("Instantiating GRN object.", flush = True)
    grn = ggrn.GRN(
        train                = train_data, 
        network              = networks[conditions.loc[i,'network_datasets']],
        eligible_regulators  = eligible_regulators,
        feature_extraction   = conditions.loc[i,"feature_extraction"],
        validate_immediately = True
    )

    def simplify_type(x):
        """Convert a pandas dataframe into JSON serializable types.

        Args:
            x (pd.DataFrame)

        Returns:
            dict
        """
        return json.loads(x.to_json())
    print("Fitting models.", flush = True)
    grn.fit(
        method                               = conditions.loc[i,"regression_method"], 
        cell_type_labels                     = None,
        cell_type_sharing_strategy           = "identical",
        network_prior                        = conditions.loc[i,"network_prior"],
        pruning_strategy                     = conditions.loc[i,"pruning_strategy"],
        pruning_parameter                    = conditions.loc[i,"pruning_parameter"],
        predict_self                         = conditions.loc[i,"predict_self"],
        matching_method                      = conditions.loc[i,"matching_method"],
        low_dimensional_structure            = conditions.loc[i,"low_dimensional_structure"],
        low_dimensional_training             = conditions.loc[i,"low_dimensional_training"],
        prediction_timescale                 = conditions.loc[i,"prediction_timescale"],
        do_parallel = do_parallel,
        kwargs                               = {
                                                k:simplify_type(conditions.loc[i,:])[k] if k in metadata["kwargs_to_expand"] else metadata["kwargs"][k] 
                                                for k in metadata["kwargs"].keys()
                                                },
    )
    return grn

def filter_genes(expression_quantified: anndata.AnnData, num_genes: int, outputs: str) -> anndata.AnnData:
    """Filter a dataset, keeping only the top-ranked genes and the directly perturbed genes.
    The top N and perturbed genes may intersect, resulting in less than num_genes returned.
    For backwards compatibility with the DCD-FG benchmarks, we do not try to fix this.

    Args:
        expression_quantified (anndata.AnnData): _description_
        num_genes: Usually number. Expected non-numeric values are "all" or None or np.NaN, and for all those inputs, we keep all genes.

    Returns:
        anndata.AnnData: Input data, but maybe with fewer genes. 
    """
    assert "highly_variable_rank" in set(expression_quantified.var.columns)
    if num_genes is None or num_genes=="all" or np.isnan(num_genes):
        return expression_quantified

    # Perturbed genes
    targeted_genes = np.where(np.array(
        [1 if g in expression_quantified.uns["perturbed_and_measured_genes"] else 0
        for g in expression_quantified.var_names]
    ))[0]
    n_targeted = len(targeted_genes)

    # Top N minus # of perturbed genes
    try:      
        variable_genes = np.where(
            expression_quantified.var["highly_variable_rank"] < num_genes - n_targeted
        )[0]
    except:
        raise Exception(f"num_genes must act like a number w.r.t. < operator; received {num_genes}.")
    
    gene_indices = np.union1d(targeted_genes, variable_genes)
    gene_set = expression_quantified.var.index[gene_indices]
    pd.DataFrame({"genes_modeled": gene_set}).to_csv(os.path.join(outputs, "genes_modeled.csv"))
    e = expression_quantified[:, gene_set].copy()
    return e


def set_up_data_networks_conditions(metadata, amount_to_do, outputs):
    """Set up the expression data, networks, and a sample sheet for this experiment."""
    # Data, networks, experiment sheet must be generated in that order because reasons
    print("Getting data...")
    perturbed_expression_data = load_perturbations.load_perturbation(metadata["perturbation_dataset"])
    try:
        timeseries_expression_data = load_perturbations.load_perturbation(metadata["perturbation_dataset"], is_timeseries=True)
    except:
        timeseries_expression_data = None
    try:
        perturbed_expression_data = perturbed_expression_data.to_memory()
    except ValueError: #Object is already in memory.
        pass
    try:
        perturbed_expression_data.X = perturbed_expression_data.X.toarray()
    except AttributeError: #Matrix is already dense.
        pass
    elap = "expression_level_after_perturbation"
    if metadata["merge_replicates"]:
        perturbed_expression_data = averageWithinPerturbation(ad=perturbed_expression_data)
        if timeseries_expression_data is not None:
            raise ValueError("We do not currently support merging of replicates in time-series training data.")
    print("...done. Getting networks...")
    # Get networks
    networks = {}
    for netName in list(metadata["network_datasets"].keys()):
        networks = networks | get_subnets(
            netName, 
            subnets = metadata["network_datasets"][netName]["subnets"], 
            target_genes = perturbed_expression_data.var_names, 
            do_aggregate_subnets = metadata["network_datasets"][netName]["do_aggregate_subnets"]
        )
    print("...done. Expanding metadata...")
    # Lay out each set of params 
    conditions = lay_out_runs(
        networks=networks, 
        metadata=metadata,
    )
    try:
        old_conditions = pd.read_csv(os.path.join(outputs, "conditions.csv"), index_col=0)
        conditions.to_csv(        os.path.join(outputs, "new_conditions.csv") )
        conditions = pd.read_csv( os.path.join(outputs, "new_conditions.csv"), index_col=0 )
        if not conditions.equals(old_conditions):
            print(conditions)
            print(old_conditions)
            raise ValueError("Experiment layout has changed. Check diffs between conditions.csv and new_conditions.csv. If synonymous, delete conditions.csv and retry.")
    except FileNotFoundError:
        pass
    conditions.to_csv( os.path.join(outputs, "conditions.csv") )
    print("... done.")
    return perturbed_expression_data, networks, conditions, timeseries_expression_data

def doSplitsMatch(
        experiment1: str, 
        experiment2: str,
        path_to_experiments = "experiments",
        ) -> bool:
    """Check whether the same examples and genes are used for the test-set in two experiments.

    Args:
        experiment1 (str): Name of an Experiment.
        experiment2 (str): Name of another Experiment.

    Returns:
        bool: True iff the test-sets match.
    """
    t1 = sc.read_h5ad(os.path.join(path_to_experiments, experiment1, "outputs", "predictions", "0.h5ad"))
    t2 = sc.read_h5ad(os.path.join(path_to_experiments, experiment2, "outputs", "predictions", "0.h5ad"))
    if not t1.var_names.equals(t2.var_names):
        return False
    if not t1.obs_names.equals(t2.obs_names):
        return False
    for f in ["perturbation", "expression_level_after_perturbation"]:
        if not all(t1.obs[f] == t2.obs[f]):
            return False
    return True

def load_custom_test_set(type_of_split, data_split_seed) -> set:
    """Retrieve a custom test set from a user-provided file.

    Args:
        type_of_split: unless this is "custom", this function returns None.
        data_split_seed: the test set is expected to be stored in a file 'custom_test_sets/<data_split_seed>.json'.

    Returns:
        set: Custom test set
    """
    if type_of_split != 'custom':
        return None
    custom_split_file = os.path.join("custom_test_sets", f"{data_split_seed}.json")
    try:
        with open(custom_split_file, 'r') as f:
            test_set = set(json.load(f))
    except Exception as e:
        raise Exception(f"Test data must be a JSON list in the file {custom_split_file}. Encountered error reading file: {repr(e)}")
    assert type(test_set)==set, f"Test data must be a set in JSON format in the file {custom_split_file}. Got type ({type(test_set)})."
    return test_set

def splitDataWrapper(
    perturbed_expression_data: anndata.AnnData,
    desired_heldout_fraction: float, 
    networks: dict, 
    allowed_regulators_vs_network_regulators: str, 
    type_of_split: str = "interventional" ,
    data_split_seed: int = None,
    verbose: bool = True,
) -> tuple:
    """Split the data into train and test.

    Args:
        - networks (dict): dict containing LightNetwork objects from the load_networks module. Used to restrict what is allowed in the test set.
        - perturbed_expression_data (anndata.AnnData): expression dataset to split
        - desired_heldout_fraction (float): number between 0 and 1. fraction in test set.
        - allowed_regulators_vs_network_regulators (str, optional): "all", "union", or "intersection".
            - If "all" (default), then any perturbation can go in the test set.
            - If "union", then perturbed genes must be regulators in at least one of the provided networks to go in the test set.
            - If "intersection", then perturbed genes must be regulators in all of the provided networks to go in the test set.
        - type_of_split (str): 
            - if "interventional" (default), then any perturbation is placed in either the training or the test set, but not both. 
            - If "simple", then we use a simple random split, and replicates of the same perturbation are allowed to go into different folds.
            - If "genetic_interaction", we put single perturbations and controls in the training set, and multiple perturbations in the test set.
            - If "demultiplexing", we put multiple perturbations and controls in the training set, and single perturbations in the test set.
            - If "stratified", we put some samples from each perturbation in the training set, and if there is replication, we put some in the test set. 
            - If "custom", we load the test set from the file 'custom_test_sets/<data_split_seed>.json'.
        - data_split_seed (int, optional): random seed. Defaults to 0, even if None is passed in. 
        - verbose (bool, optional): print split sizes?

    Returns:
        tuple of anndata objects: train, test
    """

    if data_split_seed is None:
        data_split_seed = 0

    custom_test_set = load_custom_test_set(type_of_split, data_split_seed)

    # Allow test set to only have e.g. regulators present in at least one network
    allowedRegulators = set(perturbed_expression_data.var_names)
    if allowed_regulators_vs_network_regulators == "all":
        pass
    elif allowed_regulators_vs_network_regulators == "union":
        network_regulators = set.union(*[networks[key].get_all_regulators() for key in networks])
        allowedRegulators = allowedRegulators.intersection(network_regulators)
    elif allowed_regulators_vs_network_regulators == "intersection":
        network_regulators = set.intersection(*[networks[key].get_all_regulators() for key in networks])
        allowedRegulators = allowedRegulators.intersection(network_regulators)
    else:
        raise ValueError(f"allowedRegulators currently only allows 'union' or 'all' or 'intersection'; got {allowedRegulators}")
    
    if custom_test_set is not None:
        assert type_of_split=="custom", "custom_test_set must be None unless type_of_split=='custom'."

    perturbed_expression_data_train, perturbed_expression_data_heldout = \
        _splitDataHelper(
            perturbed_expression_data, 
            allowedRegulators, 
            desired_heldout_fraction = desired_heldout_fraction,
            type_of_split            = type_of_split,
            data_split_seed = data_split_seed,
            verbose = verbose,
            custom_test_set = custom_test_set,
        )
    return perturbed_expression_data_train, perturbed_expression_data_heldout

def _splitDataHelper(adata: anndata.AnnData, 
                     allowedRegulators: list, 
                     desired_heldout_fraction: float, 
                     type_of_split: str, 
                     data_split_seed: int, 
                     verbose: bool, 
                     custom_test_set: set):
    """Determine a train-test split satisfying constraints imposed by base networks and available data.

    parameters:

    - adata (anndata.AnnData): Object satisfying the expectations outlined in the accompanying collection of perturbation data.
    - allowedRegulators (list or set): interventions allowed to be in the test set. 
    - type_of_split (str): 
        - if "interventional" (default), then any perturbation is placed in either the training or the test set, but not both. Replicates of the same perturbation always go into the same folds. All controls go in the training set. 
        - If "simple", then we use a simple random split (well, not quite: all controls go in the training set). Replicates of the same perturbation are allowed to go into different folds, or the same fold.  
        - If "genetic_interaction", we put single perturbations and controls in the training set, and multiple perturbations in the test set.
        - If "demultiplexing", we put multiple perturbations and controls in the training set, and single perturbations in the test set.
        - If "stratified", we put some samples from each perturbation in the training set, and for any that have replication, we put some in the test set. 
        - If "custom", we use custom_test_set as the test set.
    - data_split_seed (int): Used to seed the RNG. This function is deterministic and you must alter the data_split_seed if you want different random splits. 
    - verbose (bool): print train and test sizes?
    - custom_test_set: subset of perturbed_expression_data.obs_names, as a Python set. 

    Details: 

    A few factors complicate the "interventional" type of training-test split. 

    - Perturbed genes may be absent from most base GRN's due to lack of motif information or ChIP data. 
        These perhaps should be excluded from the test data to avoid obvious failure cases.
    - Perturbed genes may not be measured. These perhaps should be excluded from test data because we can't
        reasonably separate their direct vs indirect effects.

    If type_of_split=="interventional", the `allowedRegulators` arg can be specified in order to keep any user-specified
    problem cases out of the test data. No matter what, we still use those perturbed profiles as training data, hoping 
    they will provide useful info about attainable cell states and downstream causal effects. But observations may only 
    go into the test set if the perturbed genes are in allowedRegulators.

    For some values of allowedRegulators (especially intersections of many prior networks), there are many factors 
    ineligible for use as test data -- so many that we don't have enough for the test set. In this case we issue a 
    warning and assign as many as possible to test. For other cases, we have more flexibility, so in order to achieve the 
    train/test size specified by the user, we send some perturbations to the training set at random even if we would be able to 
    use them in the test set.

    If type_of_split is not "interventional", we make no provision for dealing with the above concerns.

    """
    assert type(allowedRegulators) is list or type(allowedRegulators) is set, "allowedRegulators must be a list or set of allowed genes"
    if data_split_seed is None:
        data_split_seed = 0
    # For a deterministic result when downsampling an iterable, setting a seed alone is not enough.
    # Must also avoid the use of sets. 
    if type_of_split == "interventional":
        get_unique_keep_order = lambda x: list(dict.fromkeys(x))
        allowedRegulators = [p for p in allowedRegulators if p in adata.uns["perturbed_and_measured_genes"]]
        testSetEligible   = [p for p in adata.obs["perturbation"] if     all(g in allowedRegulators for g in p.split(","))]
        testSetIneligible = [p for p in adata.obs["perturbation"] if not all(g in allowedRegulators for g in p.split(","))]
        allowedRegulators = get_unique_keep_order(allowedRegulators)
        testSetEligible   = get_unique_keep_order(testSetEligible)
        testSetIneligible = get_unique_keep_order(testSetIneligible)
        total_num_perts = len(testSetEligible) + len(testSetIneligible)
        eligible_heldout_fraction = len(testSetEligible)/(0.0+total_num_perts)
        if eligible_heldout_fraction < desired_heldout_fraction:
            print("Not enough profiles for the desired_heldout_fraction. Will use all available.")
            testSetPerturbations = testSetEligible
            trainingSetPerturbations = testSetIneligible
        elif eligible_heldout_fraction == desired_heldout_fraction: #nailed it
            testSetPerturbations = testSetEligible
            trainingSetPerturbations = testSetIneligible
        else:
            # Plenty of perts work for either.
            # Put some back in trainset to get the right size, even though we could use them in test set.
            numExcessTestEligible = int(np.ceil((eligible_heldout_fraction - desired_heldout_fraction)*total_num_perts))
            excessTestEligible = np.random.default_rng(seed=data_split_seed).choice(
                testSetEligible, 
                numExcessTestEligible, 
                replace = False)
            testSetPerturbations = [p for p in testSetEligible if p not in excessTestEligible]                      
            trainingSetPerturbations = list(testSetIneligible) + list(excessTestEligible) 
        adata_train    = adata[adata.obs["perturbation"].isin(trainingSetPerturbations),:].copy()
        adata_heldout  = adata[adata.obs["perturbation"].isin(testSetPerturbations),    :].copy()
    elif type_of_split == "simple":
        np.random.seed(data_split_seed)
        train_obs = np.random.choice(
            replace=False, 
            a = adata.obs_names, 
            size = round(adata.shape[0]*(1-desired_heldout_fraction)), 
        )
        control_obs = set(adata.obs.loc[adata.obs["is_control"],:].index)
        train_obs = np.sort(list(set(train_obs).union(control_obs)))
        test_obs = np.sort(list(set(adata.obs_names).difference(train_obs)))
        adata_train    = adata[train_obs,:].copy()
        adata_heldout  = adata[test_obs,:].copy()
    elif type_of_split == "genetic_interaction":
        adata.obs["number_of_genes_perturbed"] = [1+p.count(",") for p in adata.obs["perturbation"]]
        adata.obs["number_of_genes_perturbed"] = adata.obs["number_of_genes_perturbed"]*(1-adata.obs["is_control_int"])
        assert any( adata.obs["number_of_genes_perturbed"] > 1), "No samples have multiple perturbations so we cannot study genetic interactions."
        adata_train   = adata[adata.obs["number_of_genes_perturbed"]<=1,].copy()
        adata_heldout = adata[adata.obs["number_of_genes_perturbed"]> 1,].copy()
    elif type_of_split == "demultiplexing":
        adata.obs["number_of_genes_perturbed"] = [1+p.count(",") for p in adata.obs["perturbation"]]
        adata.obs["number_of_genes_perturbed"] = adata.obs["number_of_genes_perturbed"]*(1-adata.obs["is_control_int"])
        assert any( adata.obs["number_of_genes_perturbed"] > 1), "No samples have multiple perturbations so we cannot study demultiplexing."
        assert any( adata.obs["number_of_genes_perturbed"] == 1), "No samples have just one perturbation so we cannot study demultiplexing."
        adata_train   = adata[adata.obs["number_of_genes_perturbed"]!=1,].copy()
        adata_heldout = adata[adata.obs["number_of_genes_perturbed"]==1,].copy()
    elif type_of_split == "stratified":
        test_samples = []
        train_samples = []
        for p in adata.obs["perturbation"].unique():
            samples_this_pert = adata.obs.index[adata.obs["perturbation"]==p]
            n_train = int(np.round(np.max([1, (1-desired_heldout_fraction)*len(samples_this_pert)])))
            train_samples_this_pert = np.random.default_rng(seed=data_split_seed).choice(samples_this_pert, n_train, replace = False)
            train_samples = train_samples + list(train_samples_this_pert)
            test_samples = test_samples + list(set(samples_this_pert).difference(train_samples_this_pert))
        adata_train   = adata[train_samples, ].copy()
        adata_heldout = adata[test_samples, ].copy()        
    elif type_of_split == "custom":
        adata_train   = adata[list(set(adata.obs_names).difference(custom_test_set)), ].copy()
        adata_heldout = adata[list(custom_test_set), ].copy()
    else:
        raise ValueError(f"`type_of_split` must be one of ['simple', 'interventional', 'genetic_interaction', 'demultiplexing', 'stratified', 'custom']; got {type_of_split}.")
    
    trainingSetPerturbations = set(  adata_train.obs["perturbation"].unique())
    testSetPerturbations     = set(adata_heldout.obs["perturbation"].unique())
    adata_train.uns[  "perturbed_and_measured_genes"]     = set(adata_train.uns[  "perturbed_and_measured_genes"]).intersection(trainingSetPerturbations)
    adata_heldout.uns["perturbed_and_measured_genes"]     = set(adata_heldout.uns["perturbed_and_measured_genes"]).intersection(testSetPerturbations)
    adata_train.uns[  "perturbed_but_not_measured_genes"] = set(adata_train.uns[  "perturbed_but_not_measured_genes"]).intersection(trainingSetPerturbations)
    adata_heldout.uns["perturbed_but_not_measured_genes"] = set(adata_heldout.uns["perturbed_but_not_measured_genes"]).intersection(testSetPerturbations)
    if verbose:
        print(
            pd.DataFrame(
                {
                    "Num obs": [adata_train.n_obs, adata_heldout.n_obs],
                    "Num perts": [len(trainingSetPerturbations), len(testSetPerturbations)],
                }, 
                index = ["Train", "Test"], 
            )
        )
    return adata_train, adata_heldout


def averageWithinPerturbation(ad: anndata.AnnData, confounders = []):
    """Average the expression levels within each level of ad.obs["perturbation"].

    Args:
        ad (anndata.AnnData): Object conforming to the validity checks in the load_perturbations module.
        confounders: other factors to consider when averaging. For instance, you may want to stratify by cell cycle phase. 
    """
    if len(confounders) != 0:
        raise NotImplementedError("Haven't yet decided how to handle confounders when merging replicates.")

    perts = ad.obs["perturbation"].unique()
    new_ad = anndata.AnnData(
        X = np.zeros((len(perts), len(ad.var_names))),
        obs = pd.DataFrame(
            {"perturbation":perts}, 
            index = perts, 
            columns=ad.obs.columns.copy(),
        ),
        var = ad.var,
        dtype = np.float32
    )
    for p in perts:
        p_idx = ad.obs["perturbation"]==p
        new_ad[p,].X = ad[p_idx,:].X.mean(0)
        new_ad.obs.loc[p,:] = ad[p_idx,:].obs.iloc[0,:]
        try:
            new_ad.obs.loc[p,"expression_level_after_perturbation"] = ad.obs.loc[p_idx, "expression_level_after_perturbation"].mean()
        except:
            # If it's a multi-gene perturbation in the format "0,0,0", don't bother averaging
            # Hope to fix this eventually to average within each coord. 
            new_ad.obs.loc[p,"expression_level_after_perturbation"] = ad.obs.loc[p_idx, "expression_level_after_perturbation"][0]
    new_ad.obs = new_ad.obs.astype(dtype = {c:ad.obs.dtypes[c] for c in new_ad.obs.columns}, copy = True)
    new_ad.raw = ad.copy()
    new_ad.uns = ad.uns.copy()
    return new_ad

def train_classifier(adata: anndata.AnnData, target_key: str = None):
    """Train a random forest classifier to predict the target key. By default, it looks for "louvain".

    Args:
        adata (anndata.AnnData): perturbation transcriptomics data conforming to the expectations enforced by load_perturbations.check_perturbation_dataset(). 
        target_key (str, optional): Column of adata.obs to use as labels. Defaults to None.

    Errors:
        AssertionError if target_key is not found as a column in adata.obs
    
    Returns:
        None (if 'louvain' is not found) or a scikit learn random forest classifier.
    """
    assert target_key is None or target_key in adata.obs.columns, f"The column {target_key} was not found in the sample metadata, so the classifier could not be trained."
    if target_key is None:
        if "louvain" in adata.obs.columns:
            target_key = "louvain"
        else:
            return None
    X = adata.X
    y = adata.obs[target_key]
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X, y)
    return model

def downsample(adata: anndata.AnnData, proportion: float, seed = None, proportion_genes = 1):
    """Downsample training data to a given fraction, always keeping controls. 
    Args:
        adata (anndata.AnnData): _description_
        proportion (float): fraction of observations to keep. You may end up with a little extra because all controls are kept.
        proportion_genes (float): fraction of cells to keep. You may end up with a little extra because all perturbed genes are kept.
        seed (_type_, optional): RNG seed. Seed defaults to proportion so if you ask for 80% of cells, you get the same 80% every time.

    Returns:
        anndata.AnnData: Subsampled data.
    """
    if seed is None:
        seed = proportion
    np.random.seed(int(np.round(seed)))
    mask       = np.random.choice(a=[True, False], size=adata.obs.shape[0], p=[proportion,       1-proportion], replace = True)
    mask_genes = np.random.choice(a=[True, False], size=adata.var.shape[0], p=[proportion_genes, 1-proportion_genes], replace = True)
    adata = adata[adata.obs["is_control"] | mask, :].copy()
    perturbed_genes_remaining = set(adata.obs["perturbation"])
    adata = adata[:, [adata.var.index.isin(perturbed_genes_remaining)] | mask_genes].copy()
    print(adata.obs.shape)
    adata.uns["perturbed_but_not_measured_genes"] = set(adata.obs["perturbation"]).difference(  set(adata.var_names))
    adata.uns["perturbed_and_measured_genes"]     = set(adata.obs["perturbation"]).intersection(set(adata.var_names))
    return adata

def safe_save_adata(adata, h5ad):
    """Clean up an AnnData for saving. AnnData has trouble saving sets, pandas columns that have dtype "object", and certain matrix types."""
    adata.raw = None
    if type(adata.X) not in {np.ndarray, np.matrix, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix}:
        adata.X = scipy.sparse.csr_matrix(adata.X)
    try:
        del adata.obs["is_control"] 
    except KeyError as e:
        pass
    try:
        del adata.obs["is_treatment"] 
    except KeyError as e:
        pass
    try:
        adata.obs["expression_level_after_perturbation"] = adata.obs["expression_level_after_perturbation"].astype(str)
    except KeyError as e:
        pass
    try:
        adata.uns["perturbed_and_measured_genes"]     = list(adata.uns["perturbed_and_measured_genes"])
    except KeyError as e:
        pass
    try:
        adata.uns["perturbed_but_not_measured_genes"] = list(adata.uns["perturbed_but_not_measured_genes"])
    except KeyError as e:
        pass
    adata.write_h5ad( h5ad )

def load_successful_conditions(outputs):
    """Load a subset of conditions.csv for which predictions were successfully made."""
    conditions =     pd.read_csv( os.path.join(outputs, "conditions.csv") )
    def has_predictions(i):
        print(f"Checking for {i}", flush=True)
        try:
            X = sc.read_h5ad( os.path.join(outputs, "predictions",   str(i) + ".h5ad" ) )
            del X
            return True
        except:
            print(f"Skipping {i}: predictions could not be read.", flush = True)
            return False
    conditions = conditions.loc[[i for i in conditions.index if has_predictions(i)], :]
    return conditions

def get_subnets(netName:str, subnets:list = [], target_genes = None, do_aggregate_subnets = False) -> dict:
    """Get gene regulatory networks, possibly aggregating sub-networks and possibly including non-informative (empty, fully connected, or random) networks.
    This function is somewhat redundant with load_grn_all_subnetworks and load_grn_by_subnetwork, but it is better tailored for our benchmarking framework.

    Args:
        netName (str): Name of network to pull from collection, or "dense" or e.g. "random0.123" for random with density 12.3%. 
        subnets (list, optional): List of cell type- or tissue-specific subnetworks to include. 
        do_aggregate_subnets (bool, optional): If True, the returned dict has just one network named netName. If False,
            then the return value contains many separate networks named like netName + " " + subnet_name.

    Returns:
        dict: A dict containing LightNetwork objects representing gene regulatory networks. 
    """
    print("Getting network '" + netName + "'")
    gc.collect()
    if "random" in netName:
        networks = { 
            netName: load_networks.LightNetwork(
                df = load_networks.pivotNetworkWideToLong( 
                    load_networks.makeRandomNetwork( target_genes = target_genes, density = float( netName[6:] ) ) 
                ) 
            )
        }
    elif "empty" == netName or "dense" == netName:
        networks = { 
            netName: load_networks.LightNetwork(df=pd.DataFrame(index=[], columns=["regulator", "target", "weight"]))
        }
        if "dense"==netName:
            print("WARNING: for 'dense' network, returning an empty network. In GRN.fit(), use network_prior='ignore'. ")
    else:            
        networks = {}
        if do_aggregate_subnets:
            new_key = netName 
            if len(subnets)==0:
                subnets = load_networks.list_subnetworks(netName)
            networks[new_key] = load_networks.LightNetwork(netName, subnets)
        else:
            if len(subnets)==0:
                subnets = load_networks.list_subnetworks(netName)
            for subnet_name in subnets:
                new_key = netName + " " + subnet_name
                networks[new_key] = load_networks.LightNetwork(netName, [subnet_name])
    return networks