import unittest
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
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
                heldout     = {"demo": test_expression}, 
                predictions = {"demo": test_expression}, 
                baseline    = {"demo": test_expression["Control",:]}, 
                experiments = pd.DataFrame(["demo"], ["demo"], ["demo"]),
                outputs = "temp", 
                factor_varied = "demo",
                default_level = "demo", 
                classifier = None
            )
        )
        shutil.rmtree("temp")
    
if __name__ == '__main__':
    unittest.main()

# # If we ever want to achieve 100% test coverage, here's all the functions; good luck!
# makeMainPlots
# addGeneMetadata
# plotOneTargetGene
# postprocessEvaluations
# evaluateCausalModel
# safe_squeeze
# evaluate_per_target
# evaluate_across_targets
# evaluate_per_pert
# is_constant
# mse_top_n
# evaluate_across_perts
# evaluateOnePrediction
