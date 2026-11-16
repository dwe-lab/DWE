import pytest
import torch

from dwe._pointwise import forward_residual, ordered_parts, reverse_residual


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fusion")
@pytest.mark.parametrize("shape", [(17, 257), (256, 151936)])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_fused_preparation_preserves_values(shape, temperature):
    torch.manual_seed(71)
    probability = torch.rand(shape, device="cuda")
    teacher = torch.rand_like(probability)
    ratio = torch.randn_like(probability)
    value = (probability * ratio).sum(-1)
    expected = temperature * probability * (ratio - value.unsqueeze(-1))
    actual = reverse_residual(probability, ratio, value, temperature)
    assert torch.equal(actual, expected)
    assert torch.equal(
        forward_residual(probability, teacher, temperature),
        temperature * (probability - teacher),
    )

    # Zero and tiny rows exercise both sides without adding a mass threshold.
    ratio[0].zero_()
    ratio[1] *= 1e-30
    original = ratio.clone()
    for order in (None, torch.randperm(shape[1], device="cuda")):
        reference = ratio if order is None else ratio.index_select(1, order)
        positive, negative = ordered_parts(ratio, order)
        assert torch.equal(positive, reference.clamp_min(0))
        assert torch.equal(negative, (-reference).clamp_min(0))
    assert torch.equal(ratio, original)
