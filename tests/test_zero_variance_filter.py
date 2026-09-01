import os
import sys
import tempfile
import unittest

import joblib
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for relative_path in ("features", "methods", "train", "utils"):
    sys.path.insert(0, os.path.join(REPO_ROOT, relative_path))

from cv_baselines import fit_predict_multitask_cv, fit_zero_variance_filter


class ZeroVarianceFingerprintFilterTests(unittest.TestCase):
    def setUp(self):
        # Bits 1 and 2 are constant in train, even though they vary elsewhere.
        self.X_train = np.asarray(
            [
                [0, 1, 0, 0],
                [1, 1, 0, 1],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [1, 1, 0, 1],
            ],
            dtype=np.float32,
        )
        self.X_val = np.asarray(
            [[0, 0, 1, 1], [1, 1, 0, 0]],
            dtype=np.float32,
        )
        self.X_test = np.asarray(
            [[1, 0, 1, 0], [0, 1, 0, 1]],
            dtype=np.float32,
        )

    def test_filter_is_fit_only_on_train_and_reused_for_other_splits(self):
        X_train, X_val, X_test, bit_filter = fit_zero_variance_filter(
            self.X_train,
            self.X_val,
            self.X_test,
        )

        np.testing.assert_array_equal(
            bit_filter.get_support(),
            np.asarray([True, False, False, True]),
        )
        self.assertEqual(X_train.shape, (6, 2))
        np.testing.assert_array_equal(X_val, self.X_val[:, [0, 3]])
        np.testing.assert_array_equal(X_test, self.X_test[:, [0, 3]])

    def test_serialized_fold_model_contains_the_train_bit_filter(self):
        y_train = np.asarray([[0], [1], [0], [1], [0], [1]], dtype=np.float32)
        y_val = np.asarray([[0], [1]], dtype=np.float32)
        y_test = np.asarray([[1], [0]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = os.path.join(temp_dir, "models")
            params_dir = os.path.join(temp_dir, "params")
            result = fit_predict_multitask_cv(
                method="rf",
                X_train=self.X_train,
                y_train=y_train,
                X_val=self.X_val,
                y_val=y_val,
                X_test=self.X_test,
                y_test=y_test,
                task_type=np.asarray([1]),
                mc_label_values=[None],
                model_out_dir=model_dir,
                params_out_dir=params_dir,
                n_splits=2,
                seed=42,
            )

            filter_info = result["metadata"]["fingerprint_filter"]
            self.assertEqual(filter_info["original_bits"], 4)
            self.assertEqual(filter_info["retained_bits"], 2)
            self.assertEqual(filter_info["removed_bit_indices"], [1, 2])

            model_path = result["metadata"]["tasks"][0]["fold_models"][0]
            saved_model = joblib.load(model_path)
            predictions = saved_model.predict_proba(self.X_test)
            self.assertEqual(predictions.shape[0], self.X_test.shape[0])
            self.assertEqual(
                saved_model.named_steps["estimator"].n_features_in_,
                filter_info["retained_bits"],
            )


if __name__ == "__main__":
    unittest.main()
