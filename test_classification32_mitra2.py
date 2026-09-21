"""CPU contract tests; also reuse original classifier32 coverage adversaries."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import classification32_mitra2 as adapter
from test_mitra_class32_one import FakeTrainer, FakeEstimator, TORCH


def config():
    return SimpleNamespace(seed=0, hyperparams=dict(n_ensembles=32, max_epochs=0,
        max_samples_support=8192, max_samples_query=1024, precision='bfloat16', dim_output=10,
        grad_scaler_enabled=False, shuffle_classes=False, shuffle_features=False,
        use_random_transforms=False, random_mirror_x=True))


class Model:
    dim_output = 10
    use_flash_attn = False
    def __init__(self): self.parameter = SimpleNamespace(grad=None, requires_grad=True)
    def requires_grad_(self, value): self.parameter.requires_grad=value
    def register_forward_hook(self, _fn): pass
    def parameters(self): return [self.parameter]


class BaseTrainer:
    def __init__(self, cfg, model, rng):
        self.cfg, self.model, self.rng = cfg, model, rng
        self.optimizer = SimpleNamespace(step=lambda: None)
        self.preprocessor = SimpleNamespace(mirror=np.ones(2))
    def train(self, xs, ys, xv, yv):
        self.rng.rand(); np.random.rand()
        return self


class ChunkTrainer(FakeTrainer):
    def __init__(self, member, rng, support_rows, execution):
        super().__init__(member, rng, support_rows)
        self.execution, self.skip_tail, self.duplicate_chunk = execution, False, False
    def predict(self, xs, ys, xt):
        outputs = []
        for start in range(0, len(xt), 1024):
            query = xt[start:start+1024]
            if self.skip_tail and start > 0:
                outputs.append(np.zeros((len(query), 10), dtype=np.float32)); continue
            self.execution.append((self.member, start, len(query)))
            outputs.append(super().predict(xs, ys, query))
            if self.duplicate_chunk: super().predict(xs, ys, query)
        return np.concatenate(outputs)


class FullQueryEstimator(FakeEstimator):
    def __init__(self):
        super().__init__()
        self.execution, self.query_sizes = [], []
    def fit(self, xs, ys, X_val, y_val):
        selected = adapter.validation_indices(ys)
        np.testing.assert_array_equal(X_val, xs[selected])
        np.testing.assert_array_equal(y_val, ys[selected])
        self.X, self.y = xs, ys
        rng = np.random.RandomState(0)
        self.trainers = [ChunkTrainer(i, rng, len(ys), self.execution) for i in range(32)]
        for trainer in self.trainers:
            trainer.audit_validation_indices = selected.tolist()
            trainer.model.parameters = lambda: [SimpleNamespace(requires_grad=False, grad=None)]
        return self
    def predict_proba(self, xt):
        self.query_sizes.append(len(xt))
        return super().predict_proba(xt)


def fitted(query_rows=2053):
    xs = np.arange(60, dtype=np.float32).reshape(30,2)
    ys = np.tile(np.arange(3), 10)
    selected = adapter.validation_indices(ys)
    native = FullQueryEstimator().fit(xs, ys, X_val=xs[selected], y_val=ys[selected])
    return native, np.zeros((query_rows, 2), dtype=np.float32)


class Tests(unittest.TestCase):
    def test_historical_validation_subset_and_classes(self):
        y = np.array([0]*12 + [1]*5 + [2]*3)
        self.assertEqual(adapter.validation_indices(y).tolist(), list(range(8))+[12,17])

    def test_validation_rejects_more_than10_or_sparse_labels(self):
        for y in [np.arange(11), np.array([0,2]), np.array([0,0])]:
            with self.assertRaises(RuntimeError): adapter.validation_indices(y)

    def trainer(self, cfg=None):
        torch = SimpleNamespace(no_grad=nullcontext)
        cls=adapter.guard_trainer(BaseTrainer,torch)
        return cls(cfg or config(), Model(), rng=np.random.RandomState(0))

    def test_native_count_cap_and_noft_guards(self):
        for field,value in [('n_ensembles',1),('max_epochs',1),('max_samples_support',4096),('max_samples_query',512)]:
            cfg=config(); cfg.hyperparams[field]=value
            with self.assertRaises(RuntimeError): self.trainer(cfg)

    def test_freezes_parameters_and_blocks_optimizer(self):
        t=self.trainer()
        self.assertFalse(t.model.parameter.requires_grad)
        with self.assertRaisesRegex(RuntimeError,'Optimizer updates'): t.optimizer.step()
        self.assertEqual(t.audit_optimizer_steps,1)

    def test_tiny_validation_preserved_not_full_support(self):
        t=self.trainer(); x=np.arange(40).reshape(20,2); y=np.array([0]*12+[1]*8)
        selected=adapter.validation_indices(y)
        t.train(x,y,x[selected],y[selected])
        self.assertEqual(t.audit_fit_calls,1)
        self.assertEqual(t.audit_validation_indices,list(range(8))+[12])
        self.assertEqual(t.audit_optimizer_steps,0)

    def test_unexpected_full_validation_or_second_fit_rejected(self):
        x=np.arange(40).reshape(20,2); y=np.array([0]*12+[1]*8); t=self.trainer()
        with self.assertRaisesRegex(RuntimeError,'validation differs'): t.train(x,y,x,y)
        selected=adapter.validation_indices(y); t.train(x,y,x[selected],y[selected])
        with self.assertRaisesRegex(RuntimeError,'refit'): t.train(x,y,x[selected],y[selected])

    def test_wrong_model_or_seed_fails_before_loading(self):
        with self.assertRaisesRegex(RuntimeError,'Wrong model'): adapter.predict('mitra',{}, {})
        with self.assertRaisesRegex(RuntimeError,'seed0'): adapter.predict('mitra2',{}, dict(seed=42))

    def test_no_test_labels_permitted(self):
        arrays=dict(X_train=np.ones((2,1)),y_train=np.array([0,1]),X_test=np.ones((3,1)),y_test=np.zeros(3))
        with self.assertRaisesRegex(RuntimeError,'no test labels'): adapter.predict('mitra2',arrays,{})

    def test_full_query_native_member_major_coverage_with_duplicate_rows(self):
        native, query = fitted()
        probability, records = adapter.audited_full_probability(native, query)
        self.assertEqual(probability.shape, (2053,3))
        self.assertEqual(probability.dtype, np.float32)
        self.assertEqual(native.query_sizes, [2053])
        self.assertEqual(native.execution, [(i,start,count) for i in range(32)
            for start,count in [(0,1024),(1024,1024),(2048,5)]])
        self.assertTrue(all(r['all_test_rows_covered'] and r['actual_query_rows']==2053
            and r['model_forward_calls']==3 for r in records))
        self.assertTrue(all(not t.model.hooks for t in native.trainers))

    def test_native_short_query_single_forward(self):
        native, query = fitted(17)
        _, records = adapter.audited_full_probability(native, query)
        self.assertTrue(all(r['model_forward_calls']==1 for r in records))

    def test_partial_member_or_missing_tail_rejected(self):
        native, query = fitted(); native.skip_last=True
        with self.assertRaisesRegex(RuntimeError, '32 independently'): adapter.audited_full_probability(native, query)
        native, query = fitted(); native.trainers[4].skip_tail=True
        with self.assertRaisesRegex(RuntimeError, 'omitted query'): adapter.audited_full_probability(native, query)
        self.assertTrue(all(not t.model.hooks for t in native.trainers))

    def test_duplicate_forward_and_wrong_mean_rejected(self):
        native, query = fitted(); native.trainers[2].duplicate_chunk=True
        with self.assertRaisesRegex(RuntimeError, 'output shape'): adapter.audited_full_probability(native, query)
        native, query = fitted(); native.bad_softmax=True
        with self.assertRaisesRegex(RuntimeError, 'arithmetic mean'): adapter.audited_full_probability(native, query)

    def test_wrapper_uses_full_native_once_and_preserves_probability_dtype(self):
        native, query = fitted()
        audited = adapter.AuditedMitra2(native, np, TORCH).fit(native.X, native.y)
        probability, chunk = audited.predict_full(query)
        self.assertEqual(probability.dtype, np.float32)
        self.assertEqual(chunk, 1024)
        self.assertEqual(native.query_sizes, [len(query)])
        audit=audited.ensemble_audits[0]
        self.assertTrue(audit['actual32_verified'] and audit['all_test_rows_covered'])
        self.assertEqual(audit['actual_members_per_test_row'], 32)
        self.assertEqual(audit['total_model_forward_calls'], 96)
        with self.assertRaisesRegex(RuntimeError, 'full test once'): audited.predict_full(query)


if __name__=='__main__': unittest.main()
