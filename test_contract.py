import unittest
from types import SimpleNamespace
from common import COUNTS, validate_panel
from aggregate import ranks
from loop_runtime import override

class ContractTests(unittest.TestCase):
    def test_membership_partition(self):
        rows = list(range(457))
        shards = [rows[i::8] for i in range(8)]
        self.assertEqual(sorted(x for shard in shards for x in shard), rows)
        self.assertEqual(sum(COUNTS.values()), 457)
        self.assertEqual(sum(len(s) * 2 for s in shards), 914)

    def test_strict_panel(self):
        validate_panel([{'dataset': 'a', 'accuracy': '.8'}], {'a'})
        for data in [[], [{'dataset': 'a', 'accuracy': 'nan'}],
                     [{'dataset': 'a', 'accuracy': '.8', 'status': 'error'}],
                     [{'dataset': 'a', 'accuracy': '.8'}] * 2]:
            with self.assertRaises(AssertionError):
                validate_panel(data, {'a'})

    def test_tied_rank(self):
        self.assertEqual(ranks([.9, .8, .8, .7]), [1, 2.5, 2.5, 4])
        self.assertEqual(ranks([.8] * 9), [5.] * 9)

    def test_runtime_only_override(self):
        parameter = SimpleNamespace(_version=0)
        for passes in (2, 3, 4):
            encoder = SimpleNamespace(shared_depth_enabled=True, shared_depth_dataset_conditioned=True,
                                      shared_depth_rho=1., shared_depth_num_passes=2)
            model = SimpleNamespace(icl_predictor=SimpleNamespace(tf_icl=encoder),
                                    named_parameters=lambda: [('weight', parameter)])
            override(model, passes)
            self.assertEqual(encoder.shared_depth_num_passes, passes)
            self.assertEqual(parameter._version, 0)

if __name__ == '__main__':
    unittest.main()
