"""CPU-only tests for FP8 quantization, layout and error helpers."""
import unittest
import torch

from fp8_probe import (as_column_major, as_row_major, dequantize, pad_rows,
                       padding_plan, quantize_per_tensor, relative_frobenius)


class FP8ProbeHelperTests(unittest.TestCase):
    def test_per_tensor_scaling_e4m3_and_e5m2(self):
        x = torch.tensor([[-2., -1., 0., 1., 2.]], dtype=torch.float32)
        for name, dtype, maximum in (('e4m3fn', torch.float8_e4m3fn, 448.),
                                     ('e5m2', torch.float8_e5m2, 57344.)):
            result = quantize_per_tensor(x, name)
            self.assertEqual(result['q'].dtype, dtype)
            self.assertAlmostEqual(result['scale'], 2. / maximum)
            self.assertGreater(result['scale'], 0.)
            self.assertTrue(torch.isfinite(torch.tensor(result['scale'])))
            self.assertEqual(result['dtype'], str(dtype).replace('torch.', ''))
            self.assertTrue(torch.equal(dequantize(result['q'], result['scale']),
                                        result['q'].float() * result['scale']))
            self.assertEqual(result['saturation_count'], 0)
            self.assertEqual(result['nonzero_input_count'], 4)
            self.assertGreaterEqual(result['quantized_zero_fraction'], 0.)
            self.assertLessEqual(result['quantized_zero_fraction'], 1.)

    def test_all_zero_and_underflow(self):
        zero = quantize_per_tensor(torch.zeros(2, 3), 'e4m3fn')
        self.assertEqual(zero['scale'], 1.)
        self.assertEqual(int(torch.count_nonzero(zero['q'].float()).item()), 0)
        self.assertEqual(zero['quantized_zero_fraction'], 1.)
        self.assertEqual(zero['underflow_fraction'], 0.)
        tiny = quantize_per_tensor(torch.tensor([1., 1e-12]), 'e4m3fn')
        self.assertGreater(tiny['quantized_zero_count'], 0)
        self.assertGreater(tiny['underflow_count'], 0)
        self.assertEqual(tiny['nonzero_input_count'], 2)
        self.assertAlmostEqual(tiny['underflow_fraction'],
                               tiny['underflow_count'] / tiny['nonzero_input_count'])

    def test_signed_extremes_and_saturation_audit(self):
        x = torch.tensor([[-448., 448.]], dtype=torch.float32)
        result = quantize_per_tensor(x, 'e4m3fn')
        self.assertEqual(result['scale'], 1.)
        self.assertEqual(result['q'].float().tolist(), [[-448., 448.]])
        self.assertEqual(result['saturation_count'], 0)
        beyond = quantize_per_tensor(torch.tensor([[-896., 896.]], dtype=torch.float32),
                                     'e4m3fn')
        self.assertEqual(beyond['scale'], 2.)
        self.assertEqual(beyond['q'].float().tolist(), [[-448., 448.]])

    def test_nonfinite_rejected(self):
        for value in (float('inf'), float('-inf'), float('nan')):
            with self.assertRaises(ValueError):
                quantize_per_tensor(torch.tensor([value]), 'e4m3fn')

    def test_padding_and_layout_contract(self):
        expected = {1: 16, 3: 16, 7: 16, 17: 32}
        for real, padded in expected.items():
            plan = padding_plan(real)
            self.assertEqual(plan['real_M'], real)
            self.assertEqual(plan['padded_M'], padded)
            self.assertEqual(plan['padding_rows'], padded - real)
            self.assertAlmostEqual(plan['overhead_ratio'], padded / real)
        x = torch.ones(3, 4)
        padded, copied = pad_rows(x, 16)
        self.assertTrue(copied)
        self.assertEqual(tuple(padded.shape), (16, 4))
        self.assertTrue(torch.equal(padded[:3], x))
        self.assertEqual(int(torch.count_nonzero(padded[3:]).item()), 0)
        row = torch.randn(4, 6)
        view, copied = as_row_major(row)
        self.assertFalse(copied)
        self.assertEqual(view.stride(), (6, 1))
        col, copied = as_column_major(row)
        self.assertTrue(copied)
        self.assertEqual(col.stride(), (1, 4))
        self.assertTrue(torch.equal(col, row))
        transposed = row.t()
        view, copied = as_row_major(transposed)
        self.assertTrue(copied)
        self.assertEqual(view.stride(), (4, 1))

    def test_relative_frobenius_arithmetic(self):
        x = torch.zeros(3, 4)
        self.assertEqual(relative_frobenius(x, x), 0.)
        self.assertGreater(relative_frobenius(torch.ones(2, 2), torch.zeros(2, 2)), 1e11)
        self.assertAlmostEqual(relative_frobenius(torch.tensor([1., 2.]),
                                                   torch.tensor([2., 4.])),
                               0.5, places=6)
        with self.assertRaises(ValueError):
            relative_frobenius(torch.ones(2, 2), torch.ones(2, 1))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
