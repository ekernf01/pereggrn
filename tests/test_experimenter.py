import unittest
import perturbation_benchmarking_package.experimenter as experimenter
import load_perturbations
import numpy as np
import json
import os
load_perturbations.set_data_path("../perturbation_data/perturbations")
adata = load_perturbations.load_perturbation("norman")
adata = adata[:, adata.uns["perturbed_and_measured_genes"]] # slim it down to speed this up
custom_test_set = set(adata.obs_names[0:1000])
os.makedirs("custom_test_sets", exist_ok=True)
with open("custom_test_sets/0.json", "w") as f:
    json.dump(list(custom_test_set), f)

class TestDataSplit(unittest.TestCase):
    def test_splitDataWrapper(self):
        for type_of_split in [ 'custom', 'stratified', 'simple', 'interventional', 'genetic_interaction', 'demultiplexing' ]:
            print(f"Testing type_of_split={type_of_split}")
            train,   test = experimenter.splitDataWrapper(adata, 0.5, [], type_of_split = type_of_split, verbose = False)
            train2, test2 = experimenter.splitDataWrapper(adata, 0.5, [], type_of_split = type_of_split, verbose = False)
            assert len(set(train.obs.index).intersection(test.obs.index))==0, "Train and test set should not have any samples in common."
            assert adata.n_obs==len(set(train.obs.index).union(test.obs.index)), "Train and test set should together contain all samples"
            assert all( train.obs_names == train2.obs_names ), "Results should be exactly repeatable"
            assert all( test.obs_names   == test2.obs_names ), "Results should be exactly repeatable"
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

if __name__ == '__main__':
    unittest.main()
