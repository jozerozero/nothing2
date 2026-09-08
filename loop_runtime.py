"""Change only the number of shared ICL passes after strict checkpoint loading."""
import json
import os
from pathlib import Path

def override(model, passes):
    assert passes in (2, 3, 4)
    encoder = model.icl_predictor.tf_icl
    assert encoder.shared_depth_enabled and encoder.shared_depth_dataset_conditioned
    assert encoder.shared_depth_rho == 1.0
    assert encoder.shared_depth_num_passes == 2, 'checkpoint must be trained with two passes'
    before = {k: (id(p), p._version) for k, p in model.named_parameters()}
    model.shared_depth_icl_num_passes = passes
    encoder.shared_depth_num_passes = passes
    assert before == {k: (id(p), p._version) for k, p in model.named_parameters()}
    return encoder

def install():
    from tabicl import TabICLClassifier
    if getattr(TabICLClassifier, '_g5sc_inference_override_installed', False):
        return
    original = TabICLClassifier._load_model
    def load(self):
        from common import CHECKPOINT, OUTPUT, checkpoint_identity
        passes = int(os.environ['G5SC_INFERENCE_LOOPS'])
        assert passes in (3, 4)
        assert Path(self.model_path).resolve() == CHECKPOINT.resolve()
        expected = json.loads((OUTPUT / 'input_contract.json').read_text())['checkpoint_identity']
        assert checkpoint_identity() == expected, 'checkpoint changed after preparation'
        original(self)
        assert int(self.model_config_.get('shared_depth_icl_num_passes', 2)) == 2
        encoder = override(self.model_, passes)
        self.model_config_ = dict(self.model_config_, shared_depth_icl_num_passes=passes)
        # Audit the actual forward calls, not just the configuration flag.
        counts = [0] * len(encoder.blocks)
        verified = [False]
        def increment(index):
            def hook(_module, _args, _output):
                counts[index] += 1
            return hook
        for index, block in enumerate(encoder.blocks):
            block.register_forward_hook(increment(index))
        def reset(_module, _args):
            counts[:] = [0] * len(counts)
        def check(_module, _args, _output):
            assert counts == [passes] * len(counts), f'wrong realized loop counts: {counts}'
            if not verified[0]:
                print(json.dumps({'event': 'realized_loop_verified', 'passes': passes, 'block_calls': counts}), flush=True)
                verified[0] = True
        encoder.register_forward_pre_hook(reset)
        encoder.register_forward_hook(check)
        print(json.dumps({'event': 'strict_checkpoint_override', 'checkpoint': str(CHECKPOINT),
                          'training_loops': 2, 'inference_loops': passes, 'parameter_mutations': 0}), flush=True)
    TabICLClassifier._load_model = load
    TabICLClassifier._g5sc_inference_override_installed = True
