import unittest
import torch

from src.trainings.utils.anomaly_prediction_metrics import ExistenceOfAnomaly, DensityOfAnomalies, LeadTime, DiceScore


class TestAnomalyMetrics(unittest.TestCase):
    def setUp(self):
        """Set up common test data."""
        # Simple test cases
        self.batch_size = 3
        self.window_size = 4

        # Test Case 1: Perfect prediction
        self.perfect_preds = torch.tensor([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0]
        ])
        self.perfect_targets = self.perfect_preds.clone()

        # Test Case 2: Completely wrong prediction
        self.wrong_preds = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        self.wrong_targets = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0]
        ])

        # Test Case 3: Mixed predictions with probabilities
        self.prob_preds = torch.tensor([
            [0.2, 0.7, 0.3, 0.1],
            [0.1, 0.9, 0.2, 0.3],
            [0.8, 0.2, 0.1, 0.4]
        ])
        self.prob_targets = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0]
        ])

    def test_existence_of_anomaly(self):
        """Test ExistenceOfAnomaly metric."""
        metric = ExistenceOfAnomaly(threshold=0.5)

        # Test perfect predictions
        metric.update(self.perfect_preds, self.perfect_targets)
        self.assertEqual(metric.compute(), 1.0)
        metric.reset()

        # Test wrong predictions
        metric.update(self.wrong_preds, self.wrong_targets)
        self.assertLess(metric.compute(), 1.0)
        metric.reset()

        # Test probabilistic predictions
        metric.update(self.prob_preds, self.prob_targets)
        score = metric.compute()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_density_of_anomalies(self):
        """Test DensityOfAnomalies metric."""
        metric = DensityOfAnomalies()

        # Test perfect predictions
        metric.update(self.perfect_preds, self.perfect_targets)
        self.assertEqual(metric.compute(), 1.0)
        metric.reset()

        # Test wrong predictions
        metric.update(self.wrong_preds, self.wrong_targets)
        self.assertLess(metric.compute(), 1.0)
        metric.reset()

        # Test probabilistic predictions
        metric.update(self.prob_preds, self.prob_targets)
        score = metric.compute()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_lead_time(self):
        """Test LeadTime metric."""
        metric = LeadTime(threshold=0.5)

        # Test perfect predictions
        metric.update(self.perfect_preds, self.perfect_targets)
        self.assertEqual(metric.compute(), 1.0)
        metric.reset()

        # Test wrong predictions
        metric.update(self.wrong_preds, self.wrong_targets)
        score = metric.compute()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        metric.reset()

        # Test no anomalies case
        no_anomalies = torch.zeros((2, 4))
        metric.update(no_anomalies, no_anomalies)
        self.assertEqual(metric.compute(), float('inf'))

    def test_dice_score(self):
        """Test DiceScore metric."""
        metric = DiceScore(threshold=0.5)

        # Test perfect predictions
        metric.update(self.perfect_preds, self.perfect_targets)
        self.assertEqual(metric.compute(), 1.0)
        metric.reset()

        # Test wrong predictions
        metric.update(self.wrong_preds, self.wrong_targets)
        self.assertLess(metric.compute(), 1.0)
        metric.reset()

        # Test probabilistic predictions
        metric.update(self.prob_preds, self.prob_targets)
        score = metric.compute()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_invalid_shapes(self):
        """Test handling of invalid input shapes."""
        metrics = [
            ExistenceOfAnomaly(),
            DensityOfAnomalies(),
            LeadTime(),
            DiceScore()
        ]

        # Test 1D tensors
        invalid_1d = torch.tensor([0.0, 1.0, 0.0])

        # Test 3D tensors
        invalid_3d = torch.ones((2, 3, 4))

        # Test mismatched shapes
        mismatched_a = torch.ones((2, 3))
        mismatched_b = torch.ones((2, 4))

        for metric in metrics:
            with self.subTest(metric=type(metric).__name__):
                # Test 1D tensor
                with self.assertRaises(ValueError):
                    metric.update(invalid_1d, invalid_1d)

                # Test 3D tensor
                with self.assertRaises(ValueError):
                    metric.update(invalid_3d, invalid_3d)

                # Test mismatched shapes
                with self.assertRaises(ValueError):
                    metric.update(mismatched_a, mismatched_b)

    def test_edge_cases(self):
        """Test edge cases for all metrics."""
        metrics = [
            ExistenceOfAnomaly(),
            DensityOfAnomalies(),
            LeadTime(),
            DiceScore()
        ]

        # All zeros
        zeros = torch.zeros((2, 4))

        # All ones
        ones = torch.ones((2, 4))

        # Single anomaly
        single_anomaly = torch.tensor([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])

        for metric in metrics:
            with self.subTest(metric=type(metric).__name__):
                # Test all zeros
                metric.update(zeros, zeros)
                score = metric.compute()
                self.assertIsInstance(score, (float, torch.Tensor))
                metric.reset()

                # Test all ones
                metric.update(ones, ones)
                score = metric.compute()
                self.assertIsInstance(score, (float, torch.Tensor))
                metric.reset()

                # Test single anomaly
                metric.update(single_anomaly, single_anomaly)
                score = metric.compute()
                self.assertIsInstance(score, (float, torch.Tensor))
