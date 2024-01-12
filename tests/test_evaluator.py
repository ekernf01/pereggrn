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
test_expression = load_perturbations.load_perturbation("software_test")
test_perturbations = pd.DataFrame(
                    [
                        ("AATF", 0.0), # no connections
                        ("ALX3",0.0),  # not enough connections
                        ("MYOD1",0.0)  # enough connections
                ], 
                columns = ["perturbation", "expression_level_after_perturbation"]
            )
test_expression.obs["expression_level_after_perturbation"] = 0

class TestEvaluation(unittest.TestCase):

    def test_evaluateOnePrediction(self):
        self.assertIsNotNone(
            evaluator.evaluateOnePrediction(
                expression =  test_expression,
                predictedExpression = test_expression, 
                baseline = test_expression["Control",:], 
                doPlots=False, 
                outputs="demo",
                experiment_name="test",    
            )
        )

    def test_evaluateCausalModel(self):
        os.makedirs("temp", exist_ok=True)
        self.assertIsNotNone(
            evaluator.evaluateCausalModel(
                get_current_data_split = lambda i : (test_expression, test_expression),
                predicted_expression={0: test_expression},
                is_test_set=True,
                conditions = pd.DataFrame({"demo":["demo"]}, index = [0]),
                outputs = "temp", 
                path_to_accessory_data = "../../accessory_data",
            )
        )
        shutil.rmtree("temp")
    
    def test_addGeneMetadata(self):
        for genes_considered_as in ["targets", "perturbations"]:
            df = evaluator.addGeneMetadata(
                df = pd.DataFrame({"perturbation": ["NANOG"], "mae": [0]}, index = ["NANOG"]), 
                adata      = test_expression, 
                adata_test = test_expression,
                genes_considered_as = genes_considered_as, 
                path_to_accessory_data = "../../accessory_data",
            )[0]
            self.assertIn("in-degree_ANANSE_0.5", df.columns)
            self.assertIn("pLI", df.columns)

    def test_safe_squeeze(self):
        self.assertEqual(len(evaluator.safe_squeeze(np.matrix([1,1])).shape), 1)
        self.assertEqual(len(evaluator.safe_squeeze(csr_matrix([1,1])).shape), 1)

    def test_mse_top_n(self):
        self.assertEqual(evaluator.mse_top_n(observed = np.linspace(1, 100, 100), predicted = np.zeros(100), baseline = np.zeros(100), n = 1), 10000)
        self.assertEqual(evaluator.mse_top_n(observed = np.linspace(1, 100, 100), predicted = np.zeros(100), baseline = np.zeros(100), n = 2), 100**2 + 99**2)

    def test_metrics(self):
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

if __name__ == '__main__':
    unittest.main()

# # If we ever want to achieve 100% test coverage, here's the remaining functions; good luck!
# makeMainPlots
# plotOneTargetGene
# postprocessEvaluations
# evaluateCausalModel
# evaluate_per_target
# evaluate_across_targets
# evaluate_per_pert
# evaluate_across_perts
# evaluateOnePrediction
