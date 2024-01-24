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
            evaluator.METRICS["pvalue_effect_direction"](observed = np.array([1,0,1,0]), predicted = np.array([0,1,0,1]), baseline = np.zeros(4)), 
            0
        ) 
        self.assertAlmostEquals(
            evaluator.METRICS["pvalue_targets_vs_non_targets"](observed = np.array([1,0,1,0]), predicted = np.array([0,1,0,1]), baseline = np.zeros(4)), 
            0
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

    def test_evaluateCausalModel(self):
        os.makedirs("temp", exist_ok=True)
        evaluationPerPert, evaluationPerTarget = evaluator.evaluateCausalModel(
                get_current_data_split = lambda i : (test_expression, test_expression),
                predicted_expression={0: test_expression},
                is_test_set=True,
                conditions = pd.DataFrame({"demo":["demo"]}, index = [0]),
                outputs = "temp", 
                path_to_accessory_data = "../../accessory_data",
            )
        shutil.rmtree("temp")
        # These fields would all be included if the "conditions" input above had them. 
        expected_common_columns = ['index', 'condition', 'unique_id', 'nickname', 
                            'question', 'data_split_seed', 'type_of_split', 'regression_method', 
                            'num_genes', 'eligible_regulators', 'is_active', 'facet_by', 'color_by', 
                            'factor_varied', 'merge_replicates', 'perturbation_dataset', 'network_datasets', 
                            'refers_to', 'pruning_parameter', 'pruning_strategy', 'network_prior', 
                            'desired_heldout_fraction', 'starting_expression', 'feature_extraction', 
                            'control_subtype', 'predict_self', 'low_dimensional_structure', 
                            'low_dimensional_training', 'matching_method', 'prediction_timescale', 
                            'baseline_condition']
        expected_perturbation_effects_columns = ['logFCNorm2', 'pearsonCorr', 'spearmanCorr', 'logFC']
        expected_gene_metadata_columns = ['transcript', 'chr', 'n_exons', 'tx_start', 'tx_end', 'bp', 'mu_syn', 
                                          'mu_mis', 'mu_lof', 'n_syn', 'n_mis', 'n_lof', 'exp_syn', 'exp_mis', 
                                          'exp_lof', 'syn_z', 'mis_z', 'lof_z', 'pLI', 'n_cnv', 'exp_cnv', 'cnv_z']
        expected_expression_characteristics_columns = ['highly_variable', 'highly_variable_rank', 'means', 
                                                      'variances', 'variances_norm']
        
        for col in expected_gene_metadata_columns + expected_expression_characteristics_columns + expected_perturbation_effects_columns:
            self.assertIn(col, evaluationPerPert.columns,   f"Column '{col}' not found in evaluationPerPert")
        for col in expected_gene_metadata_columns + expected_expression_characteristics_columns:
            self.assertIn(col, evaluationPerTarget.columns, f"Column '{col}' not found in evaluationPerTarget")
    
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


    def test_evaluateOnePrediction(self):
        metrics, metrics_per_target = evaluator.evaluateOnePrediction(
            expression=test_expression,
            predictedExpression=test_expression,
            baseline=test_expression[0,:],
            outputs="test_outputs",
            experiment_name="test_experiment",
            doPlots=False
        )
        # Basic checks to ensure the function returns results in expected format
        self.assertIsInstance(metrics, pd.DataFrame, "Result metrics should be a DataFrame")
        self.assertIsInstance(metrics_per_target, pd.DataFrame, "Result metrics per target should be a DataFrame")

        # Expected columns in the output DataFrames
        expected_metrics_per_pert = ['perturbation'] + list(evaluator.METRICS.keys())
        expected_metrics_target = ['mse', 'mae', 'standard_deviation']
        for col in expected_metrics_per_pert:
            self.assertIn(col, metrics.columns,            f"Column '{col}' not found in metrics (evaluationPerPert)")
        for col in expected_metrics_target:
            self.assertIn(col, metrics_per_target.columns, f"Column '{col}' not found in metrics (evaluationPerTarget)")            

if __name__ == '__main__':
    unittest.main()

# # If we ever want to achieve 100% test coverage, here's the remaining functions; good luck!
# makeMainPlots
# plotOneTargetGene
# postprocessEvaluations
# evaluate_per_target
# evaluate_across_targets
# evaluate_per_pert
# evaluate_across_perts
