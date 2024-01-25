import unittest
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
import shutil

# Deal with various modules specific to this project
from perturbation_benchmarking_package import evaluator
import load_perturbations
load_perturbations.set_data_path("../../perturbation_data/perturbations")
test_expression = load_perturbations.load_perturbation("nakatake")
test_expression.obs["expression_level_after_perturbation"] = 0
test_expression_synthetic = pd.DataFrame(
                    [
                        ("AATF", 0.0), # no connections
                        ("ALX3",0.0),  # not enough connections
                        ("MYOD1",0.0)  # enough connections
                ], 
                columns = ["perturbation", "expression_level_after_perturbation"]
            )

class TestEvaluation(unittest.TestCase):

    def test_metrics(self):
        for k in evaluator.METRICS.keys():
            evaluator.METRICS[k](np.random.rand(10), np.random.rand(10), np.random.rand(10))
        self.assertAlmostEquals(
            evaluator.METRICS["pvalue_effect_direction"](observed = np.array([1,0,1,0]), predicted = np.array([1,0,1,0]), baseline = np.zeros(4)), 
            0.3173105078
        )
        self.assertAlmostEquals(
            evaluator.METRICS["pvalue_targets_vs_non_targets"](observed = np.array([1,0,1.2,0.2]), predicted = np.array([0,1,0,1]), baseline = np.zeros(4)), 
            0.0194193243
        )         
        self.assertAlmostEquals(
            evaluator.METRICS["fc_targets_vs_non_targets"](observed = np.array([1,0,1,0]), predicted = np.array([0,1,0,1]), baseline = np.zeros(4)), 
            -1
        )     
        self.assertAlmostEquals(
            evaluator.METRICS["fc_targets_vs_non_targets"](observed = np.array([0.25,0.75,0.25,0.75]), predicted = np.array([0,1,0,1]), baseline = np.zeros(4)), 
            0.5
        ) 
        self.assertAlmostEquals(
            evaluator.METRICS["mse_top_100"](observed = np.linspace(1, 100, 100), predicted = np.zeros(100), baseline = np.zeros(100)), 
            (np.linspace(1, 100, 100)**2).sum()
        )
        self.assertAlmostEquals(
            evaluator.METRICS["mse_top_100"](observed = np.linspace(1, 100, 100), predicted = np.zeros(100), baseline = np.zeros(100)), 
            (np.linspace(1, 100, 100)**2).sum()
        )
        self.assertAlmostEquals(
            evaluator.METRICS["mse"](observed = np.linspace(1, 100, 100), predicted = np.zeros(100), baseline = np.zeros(100)), 
            (np.linspace(1, 100, 100)**2).sum()
        )
        self.assertAlmostEquals(
            evaluator.METRICS["mae"](observed = np.linspace(1, 100, 100), predicted = np.zeros(100), baseline = np.zeros(100)), 
            (np.linspace(1, 100, 100)).mean()
        )
        self.assertAlmostEquals(
            evaluator.METRICS["proportion_correct_direction"](observed = np.linspace(1, 100, 100), predicted = np.zeros(100), baseline = np.zeros(100)),
            0
        )
        self.assertAlmostEquals(
            evaluator.METRICS["proportion_correct_direction"](observed = np.linspace(1, 100, 100), predicted = np.linspace(1, 100, 100), baseline = np.zeros(100)),
            1
        )
        self.assertAlmostEquals(
            evaluator.METRICS["spearman"](observed = np.linspace(1, 100, 100), predicted = np.linspace(1, 100, 100), baseline = np.zeros(100)),
            1
        )
        self.assertAlmostEquals(
            evaluator.METRICS["proportion_correct_direction"](observed = np.linspace(1, 100, 100), predicted = -np.linspace(1, 100, 100), baseline = np.zeros(100)),
            0
        )


    def test_mse_top_n(self):
        self.assertEqual(evaluator.mse_top_n(observed = np.linspace(1, 100, 100), predicted = np.zeros(100), baseline = np.zeros(100), n = 1), 10000)
        self.assertEqual(evaluator.mse_top_n(observed = np.linspace(1, 100, 100), predicted = np.zeros(100), baseline = np.zeros(100), n = 2), 100**2 + 99**2)


if __name__ == '__main__':
    unittest.main()
