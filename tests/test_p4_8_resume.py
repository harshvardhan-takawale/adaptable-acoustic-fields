"""P4-8: resuming loads checkpoints to CPU (tinycudann cannot use torch's cached GPU blocks). Behaviour must be
unchanged: a resumed trainer holds EXACTLY the saved model, optimiser and scheduler state and the saved iteration."""
from __future__ import annotations

import json

import pytest
import torch

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinycudann needs a GPU")
MIX = "configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json"


@GPU
def test_resume_restores_model_optimizer_scheduler_exactly(tmp_path):
    import yaml

    from aaf.train.multi_room_2d_mat import P32TrainCfg, P32Trainer
    m = json.load(open(MIX))
    rows = [r for r in m["configs"] if r["origin"] == "C500"][:2] + [r for r in m["configs"] if r["origin"] == "CX500"][:2]
    man = tmp_path / "man.json"
    man.write_text(json.dumps(dict(m, configs=rows)))
    base = yaml.safe_load(open("configs/sweep_2d_mat/P4_7_GX.yaml"))
    base = {k: v for k, v in base.items() if k in P32TrainCfg.__dataclass_fields__}
    base.update({"configs_manifest": str(man), "n_iters": 6, "val_max_configs": 2})
    a = P32Trainer(P32TrainCfg(**base), tmp_path / "run")
    for it in (1, 2, 3):
        a._step_fn(it)
    a.save_ckpt(3)
    b = P32Trainer(P32TrainCfg(**base), tmp_path / "run")
    assert b.start_iter == 3
    sa, sb = a.model.state_dict(), b.model.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    oa, ob = a.opt.state_dict()["state"], b.opt.state_dict()["state"]
    assert oa.keys() == ob.keys()
    for k in oa:
        for n in oa[k]:
            va, vb = oa[k][n], ob[k][n]
            assert (torch.equal(va, vb) if torch.is_tensor(va) else va == vb), (k, n)
            if torch.is_tensor(vb) and vb.dim() > 0:
                assert vb.device == next(b.model.parameters()).device      # state lands on the parameters' device
    assert a.sched.state_dict() == b.sched.state_dict()
