import unittest
import perturbation_benchmarking_package.experimenter as experimenter
import load_perturbations
import load_networks
import numpy as np
import json
import os
import shutil
load_networks.set_grn_location("../../network_collection/networks")
load_perturbations.set_data_path("../../perturbation_data/perturbations")
adata = load_perturbations.load_perturbation("norman")
adata = adata[:, adata.uns["perturbed_and_measured_genes"]] # slim it down to speed this up


class TestMetadataExpand(unittest.TestCase):
    def test_metadata_parsing(self):
        metadata = {
            "unique_id": "1.6.1_1",
            "nickname": "dcdfg",
            "readme": "Do DCD-FG and/or its variants beat simple baselines on their or our test data?",
            "question": "1.6",
            "is_active": True,
            "facet_by": "starting_expression",
            "color_by": None,
            "factor_varied": "regression_method",
            "regression_method":[
                "mean", 
                "median"
            ],
            "baseline_condition": 0,
            "merge_replicates": False,
            "perturbation_dataset": "nakatake",
            "num_genes": 1000,
            "starting_expression": ["control", "heldout"],
            "network_datasets": {
                "empty":{}
            }
        }
        metadata = experimenter.validate_metadata(metadata=metadata)
        conditions = experimenter.lay_out_runs( metadata = metadata, networks = {"empty": experimenter.get_subnets("empty")} )
        assert conditions.shape[0] == 4
        conditions = experimenter.lay_out_runs( metadata = metadata|{"expand":"ladder"}, networks = {"empty": experimenter.get_subnets("empty")} )
        assert conditions.shape[0] == 2

class TestDataSplit(unittest.TestCase):
    def test_splitDataWrapper(self):
        custom_test_set = set(adata.obs_names[0:1000])
        os.makedirs("custom_test_sets", exist_ok=True)
        with open("custom_test_sets/0.json", "w") as f:
            json.dump(list(custom_test_set), f)
        for type_of_split in [ 'custom', 'stratified', 'simple', 'interventional', 'genetic_interaction', 'demultiplexing' ]:
            print(f"Testing type_of_split={type_of_split}")
            train,   test = experimenter.splitDataWrapper(adata, 0.5, [], type_of_split = type_of_split, verbose = True)
            train2, test2 = experimenter.splitDataWrapper(adata, 0.5, [], type_of_split = type_of_split, verbose = False)
            assert len(set(train.obs.index).intersection(test.obs.index))==0, "Train and test set should not have any samples in common."
            assert all(np.sort(list(adata.obs_names))==np.sort(list(set(train.obs.index).union(test.obs.index)))), "Train and test set should together contain all samples"
            assert all( train.obs_names == train2.obs_names ), "Results should be exactly repeatable"
            assert all( test.obs_names   == test2.obs_names ), "Results should be exactly repeatable"
            assert test.n_obs>0, "Test set should not be empty."
            assert train.n_obs>0, "Train set should not be empty."
            assert any(train.obs["is_control"]), "Train set should contain at least some controls."
            if type_of_split == "custom":
                assert all(np.sort(list(custom_test_set)) == np.sort(list(test.obs_names))), "Test set should match the provided custom set."
            if type_of_split == "interventional":
                assert 0==len(set(train.obs["perturbation"]).intersection(set(test.obs["perturbation"]))), "Train and test set should not have any perturbations in common."
            if type_of_split == "stratified":
                assert 0==len(set(test.obs["perturbation"]).difference(set(train.obs["perturbation"]))), "Train set should contain all perturbations present in test set."
            if type_of_split == "genetic_interaction":
                assert all([len(p.split(","))<=1 for p in train.obs["perturbation"]]), "Train set should contain only single perturbations and controls."
                assert all([len(p.split(","))>1  for p in  test.obs["perturbation"]]), "Test set should contain only multiple perturbations."
            if type_of_split == "demultiplexing":
                assert all([ic or (len(p.split(","))>1) for p,ic in zip(train.obs["perturbation"], train.obs["is_control"])]), "Train set should contain only multiple perturbations and controls."
                assert all([1==len(p.split(",")) for p in test.obs["perturbation"]]),"Test set should contain only single perturbations."
        shutil.rmtree("custom_test_sets")

if __name__ == '__main__':
    unittest.main()

# # If we ever want to achieve 100% test coverage, here's all the functions; good luck!
# get_required_keys
# get_optional_keys
# get_default_metadata
# validate_metadata
# lay_out_runs
# do_one_run
# simplify_type
# get_subnets
# filter_genes
# set_up_data_networks_conditions
# doSplitsMatch
# load_custom_test_set
# splitDataWrapper
# _splitDataHelper
# averageWithinPerturbation
# train_classifier
# downsample
# safe_save_adata
# load_successful_conditions
# has_predictions