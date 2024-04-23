import unittest
import torch
from torch_uncertainty.metrics import InverseMAE 
from torch_uncertainty.metrics import InverseRMSE 

class TestInverseMAE(unittest.TestCase):
    def test_simple_case(self):
        preds = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        metric = InverseMAE()
        metric.update(preds, target)
        result = metric.compute()
        expected = torch.tensor(1.0) 
        torch.testing.assert_allclose(result, expected)


class TestInverseRMSE(unittest.TestCase):
    def test_inverse_rmse_simple_case(self):
        preds = torch.tensor([2.5, 0.0, 2, 8])
        target = torch.tensor([3.0, -0.5, 2, 7])
        metric = InverseRMSE()
        metric.update(preds, target)
        result = metric.compute()
        
        # Calculate the expected inverse RMSE
        mse_val = torch.mean((preds - target) ** 2)
        expected = torch.reciprocal(torch.sqrt(mse_val))
        
        torch.testing.assert_allclose(result, expected)

if __name__ == '__main__':
    unittest.main()
