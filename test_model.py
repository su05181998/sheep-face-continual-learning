import torch

from pgrcl.model import AdaLSConv, CGSFNet


def small_model(stage4_block="msa"):
    return CGSFNet(
        image_size=64,
        num_classes=4,
        embedding_dim=32,
        channels=(8, 16, 24, 32),
        depths=(0, 2, 2, 2),
        key_dims=(4, 4, 4, 4),
        num_heads=(1, 1, 1, 2),
        small_context_kernel=3,
        large_context_kernel=5,
        ska_kernel=3,
        ska_groups=2,
        stage4_block=stage4_block,
    )


def test_forward_shapes_and_normalization():
    model = small_model()
    logits, features = model(torch.randn(2, 3, 64, 64), return_features=True)
    assert logits.shape == (2, 4)
    assert features.shape == (2, 32)
    assert torch.allclose(features.norm(dim=1), torch.ones(2), atol=1e-5)


def test_adaptive_scale_starts_at_equal_fusion():
    module = AdaLSConv(8, 3, 5, 3, 2)
    assert torch.allclose(module.lkp.alpha, torch.tensor(0.5))
    assert module(torch.randn(2, 8, 8, 8)).shape == (2, 8, 8, 8)


def test_stage4_adals_ablation_runs():
    assert small_model("adals")(torch.randn(2, 3, 64, 64)).shape == (2, 4)
