import torch


def test_dropin_forward_default_tensor():
    from trix.nn import DropInFFN, DropInConfig

    torch.manual_seed(0)
    m = DropInFFN(
        DropInConfig(d_model=32, num_tiles=8, tiles_per_cluster=4), mode="dynamic"
    )
    x = torch.randn(2, 4, 32)
    y = m(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape


def test_dropin_forward_return_aux():
    from trix.nn import DropInFFN, DropInConfig

    torch.manual_seed(0)
    m = DropInFFN(
        DropInConfig(d_model=16, num_tiles=4, tiles_per_cluster=2), mode="packed"
    )
    x = torch.randn(2, 3, 16)
    out, info, aux = m(x, return_aux=True)
    assert out.shape == x.shape
    assert "routing_backend" in info
    assert info["routing_backend"] == "flat_popcount"
    assert "total_aux" in aux
