"""All training ranks validate actual local device and loop forward before DDP."""
import json
import os
from pathlib import Path
import socket

import torch


def probe(model, checkpoint_dir):
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    device = next(model.parameters()).device
    assert device.type == 'cuda'
    assert torch.cuda.device_count() == 8, torch.cuda.device_count()
    assert device.index == local_rank == torch.cuda.current_device()
    assert world == 64 and model.max_classes == 0
    enc = model.icl_predictor.tf_icl
    assert len(enc.blocks) == 12 and enc.shared_depth_num_passes in (3, 4)
    assert torch.count_nonzero(enc.shared_depth_gate) == 0
    assert torch.count_nonzero(enc.shared_depth_condition_weight) == 0
    counts = [0] * len(enc.blocks)
    handles = []
    for i, block in enumerate(enc.blocks):
        def hook(_m, _a, _o, i=i):
            counts[i] += 1
        handles.append(block.register_forward_hook(hook))
    try:
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(910043)
            x = torch.randn(1, 16, 6, device=device)
            y = torch.randn(1, 10, device=device)
            prediction = model(x, y)
            assert counts == [enc.shared_depth_num_passes]*12, counts
            forward_counts = counts.copy()
            assert prediction.shape == (1, 6, 999)
            assert prediction.dtype == torch.float32 and torch.isfinite(prediction).all()
            prediction.square().mean().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in [enc.shared_depth_gate, enc.shared_depth_condition_weight])
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    receipt = {'status':'PASS', 'rank':rank, 'local_rank':local_rank, 'world_size':world,
               'node':socket.gethostname(), 'device':str(device),
               'visible_devices':torch.cuda.device_count(), 'passes':enc.shared_depth_num_passes,
               'block_calls':forward_counts, 'dtype':'float32', 'head_quantiles':999,
               'gpu_name':torch.cuda.get_device_name(device),
               'cuda_visible':os.getenv('CUDA_VISIBLE_DEVICES'),
               'rocr_visible':os.getenv('ROCR_VISIBLE_DEVICES')}
    path = Path(checkpoint_dir)/f'preflight-rank-{rank}.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(receipt)+'\n')
    os.replace(tmp, path)
    print('T25_LOOP_PREFLIGHT '+json.dumps(receipt), flush=True)
