"""CPU-only adversarial tests. No native model/checkpoint/GPU is loaded."""
import copy
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import classification32_limix as adapter


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    @property
    def dtype(self): return self.value.dtype
    @property
    def shape(self): return self.value.shape
    @property
    def ndim(self): return self.value.ndim
    def __len__(self): return len(self.value)
    def __getitem__(self, key): return Tensor(self.value[key])
    def __truediv__(self, value): return Tensor(self.value / value)
    def float(self): return Tensor(self.value.astype(np.float32))
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def mean(self, dim): return Tensor(self.value.mean(axis=dim))
    def all(self): return self.value.all()


def softmax(value, dim):
    values = value.value
    values = np.exp(values - values.max(axis=dim, keepdims=True))
    return Tensor(values / values.sum(axis=dim, keepdims=True))


TORCH = SimpleNamespace(is_tensor=lambda x: isinstance(x, Tensor),
    isfinite=lambda x: Tensor(np.isfinite(x.value)), is_grad_enabled=lambda: False,
    stack=lambda values: Tensor(np.stack([v.value for v in values])),
    nn=SimpleNamespace(functional=SimpleNamespace(softmax=softmax)))


class Scaler:
    def fit_transform(self, value): return value.copy()
    def transform(self, value): return value.copy()


class Model:
    def __init__(self): self.hooks = []
    def register_forward_hook(self, fn, with_kwargs):
        assert with_kwargs
        self.hooks.append(fn)
        return SimpleNamespace(remove=lambda: self.hooks.remove(fn))
    def __call__(self, **kwargs):
        query = kwargs['x'].value[:, kwargs['eval_pos']:]
        logits = np.repeat(query[:, :, :1], 10, axis=-1)
        logits = logits * np.arange(1, 11, dtype=np.float32)
        out = Tensor(logits)
        for hook in self.hooks: hook(self, (), kwargs, out)
        return out


def fake_runtime(count=32, skip_row=False, duplicate_index=False, corrupt_mean=False):
    def cluster(_attention):
        indices = np.arange(5)[::-1]
        if duplicate_index: indices[-1] = indices[0]
        return ({0: [0, 1], 1: [0, 1]}, {0: indices[:2], 1: indices[2:]})
    im = SimpleNamespace(cluster_test_data=cluster)
    class Retrieval:
        def __init__(self, model, sample_selection_type):
            self.model, self.sample_selection_type = model, sample_selection_type
        def inference(self, xs, ys, xt, **kwargs):
            _support, groups = im.cluster_test_data(None)
            outputs, order = [], []
            for indices in groups.values():
                if skip_row: indices = indices[:-1]
                if not len(indices): continue
                x = Tensor(np.concatenate([xs.value, xt.value[indices]])[None])
                outputs.append(self.model(x=x, y=ys, eval_pos=len(xs), task_type='cls').value[0])
                order.extend(indices.tolist())
            result = np.zeros((len(xt), 10), dtype=np.float32)
            if order: result[order] = np.concatenate(outputs)
            return Tensor(result)
    im.InferenceResultWithRetrieval = Retrieval
    pm = SimpleNamespace(InferenceResultWithRetrieval=Retrieval)
    class Predictor:
        def __init__(self):
            self.n_estimators, self.seed = 32, 0
            self.mix_precision, self.softmax_temperature = True, 0.9
            self.mask_prediction = self.inference_with_DDP = False
            self.preprocess_num = 10
            self.preprocess_pipelines = [[] for _ in range(32)]
            rng = random.Random(0)
            self.seeds = [rng.randint(0, 10000) for _ in range(320)]
            self.all_shifts = np.arange(32)
            self.model = Model()
        def predict(self, xs, ys, xt, task_type):
            self.class_permutations = [np.arange(2)[::(-1 if i % 2 else 1)] for i in range(32)]
            values = []
            for i in range(count):
                model = pm.InferenceResultWithRetrieval(self.model, 'AM')
                raw = model.inference(Tensor(xs+i), Tensor(ys), Tensor(xt+i), task_type='cls')
                logits = (raw[:, :2].float() / self.softmax_temperature)[..., self.class_permutations[i]]
                values.append(softmax(logits, 1))
            result = TORCH.stack(values).mean(0).float().cpu().numpy()
            result /= result.sum(axis=1, keepdims=True)
            if corrupt_mean: result = result[:, ::-1]
            return result
    return Predictor(), pm, im


class Tests(unittest.TestCase):
    def setUp(self):
        self.arrays = dict(X_train=np.array([[1, 2], [3, 4]], dtype=np.float32),
                           y_train=np.array([0, 1]), X_test=np.ones((5, 2), dtype=np.float32))

    def test_inputs_forbid_test_labels(self):
        with self.assertRaisesRegex(RuntimeError, 'no test labels'):
            adapter.validate_arrays(dict(self.arrays, y_test=np.zeros(5)))

    def test_inputs_no_nonfinite_or_fractional_labels(self):
        for y in [np.array([0, .5]), np.array([0, np.nan])]:
            with self.assertRaises(RuntimeError): adapter.validate_arrays(dict(self.arrays, y_train=y))

    def test_input_copies_preserve_caller(self):
        xs, ys, xt = adapter.validate_arrays(self.arrays)
        xs[:] = 5; xt[:] = 6; ys[:] = 9
        self.assertEqual(self.arrays['X_train'][0, 0], 1)

    def test_config_ratio_and_independent_copies(self):
        base = [dict(FeatureShuffler=dict(mode='shuffle'),
                retrieval_config=dict(use_retrieval=True, use_cluster=True, subsample_type='sample',
                                      retrieval_before_preprocessing=False), marker=i) for i in range(4)]
        expanded = adapter.expand_config(base)
        self.assertEqual([c['marker'] for c in expanded], list(range(4))*8)
        expanded[0]['FeatureShuffler']['mode'] = 'changed'
        self.assertEqual(base[0]['FeatureShuffler']['mode'], 'shuffle')
        self.assertEqual(expanded[4]['FeatureShuffler']['mode'], 'shuffle')

    def test_config_rejects_noretrieval_or_partial_recipe(self):
        with self.assertRaises(RuntimeError): adapter.expand_config([{}]*3)
        with self.assertRaises(RuntimeError): adapter.expand_config([{}]*4)

    def test_query_duplicates_are_covered_by_positions(self):
        x = np.ones((5, 2), dtype=np.float32)
        cov = adapter.QueryCoverage(x)
        cov.set_clusters({0: [4, 1], 1: [0, 2, 3]})
        for ids in [[4, 1], [0], [2, 3]]:
            cov.observe(np.concatenate([x[:2], x[ids]])[None], 2, np.zeros((1, len(ids), 10)))
        self.assertEqual(cov.finish()['actual_query_rows'], 5)

    def test_query_indices_cannot_duplicate_or_omit(self):
        for ids in [[0, 1, 2, 3, 3], [0, 1, 2, 3]]:
            cov = adapter.QueryCoverage(np.ones((5, 2)))
            with self.assertRaises(RuntimeError): cov.set_clusters({0: ids})

    def test_missing_model_forward_not_accepted(self):
        cov = adapter.QueryCoverage(np.ones((5, 2)))
        cov.set_clusters({0: list(range(5))})
        with self.assertRaises(RuntimeError): cov.finish()

    def test_model_row_reordering_rejected(self):
        cov = adapter.QueryCoverage(np.arange(10).reshape(5, 2))
        cov.set_clusters({0: list(range(5))})
        with self.assertRaises(RuntimeError): cov.observe(np.ones((1, 7, 2)), 2, np.zeros((1, 5, 10)))

    def run_node(self, **options):
        predictor, pm, im = fake_runtime(**options)
        factory, cluster = pm.InferenceResultWithRetrieval, im.cluster_test_data
        with patch.dict(sys.modules, {'sklearn.preprocessing': SimpleNamespace(MinMaxScaler=Scaler)}):
            try:
                return adapter.audited_node(predictor, pm, im, self.arrays['X_train'],
                                           self.arrays['y_train'], self.arrays['X_test'], TORCH)
            finally:
                self.assertIs(pm.InferenceResultWithRetrieval, factory)
                self.assertIs(im.cluster_test_data, cluster)
                self.assertEqual(predictor.model.hooks, [])

    def test_actual32_native_mean_and_complete_coverage(self):
        probs, audit = self.run_node()
        self.assertEqual(probs.shape, (5, 2))
        self.assertTrue(audit['actual32_verified'])
        self.assertEqual(len(audit['member_audits']), 32)
        self.assertTrue(all(m['actual_model_forwards'] == 2 for m in audit['member_audits']))

    def test_nominal32_actual31_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Fewer than32'): self.run_node(count=31)

    def test_member_cannot_skip_query_computation(self):
        with self.assertRaises(RuntimeError): self.run_node(skip_row=True)

    def test_duplicate_cluster_mapping_rejected(self):
        with self.assertRaises(RuntimeError): self.run_node(duplicate_index=True)

    def test_changed_native_mean_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'mean changed'): self.run_node(corrupt_mean=True)

    def test_repeated_pipeline_rng_schedule_rejected(self):
        p, _, _ = fake_runtime()
        p.seeds = [0]*320
        with self.assertRaisesRegex(RuntimeError, 'seed schedules repeated'): adapter.pipeline_records(p)

    def test_repeated_pipeline_instances_rejected(self):
        p, _, _ = fake_runtime()
        p.preprocess_pipelines = [[]]*32
        with self.assertRaisesRegex(RuntimeError, 'pipeline objects'): adapter.pipeline_records(p)


if __name__ == '__main__': unittest.main()
