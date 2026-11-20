"""Train/eval mode in kb_eval, save/load and Lightning."""
import pytest
import torch
import torch.nn as nn

import sltn


def _sig():
    sig = sltn.Signature("s")
    sig.sort("P")
    sig.variable("x", "P")
    sig.constant("c", "P")
    sig.predicate("A", ["P"])
    sig.predicate("B", ["P"])
    return sig


def _interp(sig=None, dropout=0.0, seed=0):
    torch.manual_seed(seed)
    sig = sig or _sig()
    interp = sltn.Interpretation(sig)
    interp["P"] = sltn.Type("P", shape=(3,))
    interp["A"] = nn.Sequential(nn.Linear(3, 8), nn.Dropout(dropout), nn.Linear(8, 1), nn.Sigmoid())
    interp["B"] = lambda v: torch.sigmoid(v.sum(-1, keepdim=True))
    interp["c"] = nn.Parameter(torch.randn(3))
    interp["x"] = torch.randn(16, 3)
    return interp


# --- kb_eval -------------------------------------------------------------------

@pytest.mark.parametrize("start_training", [True, False])
def test_kb_eval_uses_eval_mode_and_restores_mode(start_training):
    interp = _interp(dropout=0.9)
    kb = dict(phi=interp.signature.parse("forall x: A(x)"))
    interp.train(start_training)
    vals = [sltn.kb_eval(interp, kb)["phi"] for _ in range(5)]
    assert len(set(vals)) == 1                     # Dropout off -> deterministic
    assert interp.training is start_training
    assert interp["A"].training is start_training
    interp.eval()
    with torch.no_grad():
        assert interp(kb["phi"]).value.item() == pytest.approx(vals[0])


def test_kb_eval_restores_mode_on_error():
    interp = _interp()
    interp.train()
    bad = {"phi": interp.signature.parse("forall x: A(x)")}
    interp["A"] = nn.Sequential(nn.Linear(4, 1))   # wrong input size -> error
    with pytest.raises(RuntimeError):
        sltn.kb_eval(interp, bad)
    assert interp.training


# --- state_dict keys -------------------------------------------------------------

def test_state_dict_holds_only_registered_groundings():
    interp = _interp()
    assert list(interp.state_dict()) == ["_grounded.g_c", "_grounded.g_A.0.weight", "_grounded.g_A.0.bias",
                                         "_grounded.g_A.2.weight", "_grounded.g_A.2.bias"]
    assert [n for n, _ in interp.named_parameters()] == ["c", "A.0.weight", "A.0.bias",
                                                         "A.2.weight", "A.2.bias"]


def test_state_dict_round_trip():
    src, dst = _interp(seed=0), _interp(seed=1)
    dst.load_state_dict(src.state_dict())
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)


def test_state_dict_nested_in_module():
    class Wrapper(nn.Module):
        def __init__(self, interp):
            super().__init__()
            self.interp = interp
            self.head = nn.Linear(2, 2)

    a, b = Wrapper(_interp(seed=0)), Wrapper(_interp(seed=1))
    keys = list(a.state_dict())
    assert "interp._grounded.g_A.0.weight" in keys and "head.weight" in keys
    b.load_state_dict(a.state_dict())
    assert torch.equal(b.interp["c"], a.interp["c"])


def test_load_state_dict_reports_unknown_symbol():
    interp = _interp()
    sd = dict(interp.state_dict())
    sd["_grounded.g_Z.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="g_Z.weight"):
        interp.load_state_dict(sd)


# --- save / load with plain PyTorch -----------------------------------------------------

def test_torch_save_load_round_trip(tmp_path):
    src = _interp(seed=0)
    phi = src.signature.parse("forall x: A(x) & B(c)")
    torch.save(src.state_dict(), tmp_path / "i.pt")
    dst = _interp(seed=1)
    dst["x"] = src["x"]
    dst.load_state_dict(torch.load(tmp_path / "i.pt", weights_only=True))
    src.eval(), dst.eval()
    assert dst(phi).value.item() == pytest.approx(src(phi).value.item())


def test_load_state_dict_names_missing_symbol(tmp_path):
    src = _interp()
    dst = _interp()
    dst["A"] = lambda v: torch.sigmoid(v.sum(-1, keepdim=True))   # A no longer a module
    with pytest.raises(RuntimeError, match=r"Unexpected key\(s\).*A\.0\.weight"):
        dst.load_state_dict(src.state_dict())


def test_save_load_after_training_and_deepcopy(tmp_path):
    interp = _interp()
    kb = {"phi": interp.signature.parse("forall x: A(x)")}
    opt = torch.optim.Adam(interp.parameters(), lr=0.05)
    for _ in range(20):
        opt.zero_grad()
        sltn.kb_backward(opt, interp, kb)
        opt.step()
    torch.save(interp.state_dict(), tmp_path / "i.pt")
    other = _interp(seed=3)
    other["x"] = interp["x"]
    other.load_state_dict(torch.load(tmp_path / "i.pt", weights_only=True))
    assert sltn.kb_eval(other, kb)["phi"] == pytest.approx(sltn.kb_eval(interp, kb)["phi"])
    assert list(interp.deepcopy().state_dict()) == list(interp.state_dict())


# --- Lightning ---------------------------------------------------------------------------

def test_lightning_trains_interpretation(tmp_path):
    L = pytest.importorskip("lightning")
    interp = _interp()
    kb = {"phi": interp.signature.parse("forall x: A(x)")}
    data = torch.utils.data.DataLoader(torch.randn(64, 3), batch_size=16)

    class KBModule(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.interp = interp

        def training_step(self, batch, _):
            self.interp["x"] = batch
            return 1 - self.interp(kb["phi"]).value

        def configure_optimizers(self):
            return torch.optim.Adam(self.interp.parameters(), lr=0.05)

    before = sltn.kb_eval(interp, kb)["phi"]
    trainer = L.Trainer(max_epochs=5, accelerator="cpu", logger=False,
                        enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False, default_root_dir=tmp_path)
    trainer.fit(KBModule(), data)
    assert sltn.kb_eval(interp, kb)["phi"] > before
