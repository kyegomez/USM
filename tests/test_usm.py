import torch
import pytest
from usm_torch import USM

# Basic functionality tests
def test_forward_pass():
    model = USM(
        dim=80,
        heads=4,
        ff_dim=256,
        depth=4,
        depthwise_conv_kernel_size=31,
        dropout=0.1,
        use_group_norm=True,
        conv_first=True
    )
    input = torch.randn(10, 80, 100)
    lengths = torch.randint(10, 100, (10,))
    output, _ = model(input, lengths)
    assert output.shape == (10, 80, 100)

# Test different dropout rates
@pytest.mark.parametrize("dropout", [0.0, 0.1, 0.5])
def test_dropout_effect(dropout):
    model = USM(
        dim=80,
        heads=4,
        ff_dim=256,
        depth=4,
        depthwise_conv_kernel_size=31,
        dropout=dropout,
        use_group_norm=True,
        conv_first=True
    )
    input = torch.randn(10, 80, 100)
    lengths = torch.randint(10, 100, (10,))
    output, _ = model(input, lengths)
    assert output.shape == (10, 80, 100)

# Test invalid input shape
def test_invalid_input_shape():
    model = USM(
        dim=80,
        heads=4,
        ff_dim=256,
        depth=4,
        depthwise_conv_kernel_size=31,
        dropout=0.1,
        use_group_norm=True,
        conv_first=True
    )
    # Input shape should be (batch, dim, seq_len)
    input = torch.randn(10, 80, 1000)  # Invalid shape
    lengths = torch.randint(10, 1000, (10,))
    with pytest.raises(ValueError):
        model(input, lengths)

# Test the effect of changing depth
def test_different_depth():
    depths = [2, 4, 6]
    for depth in depths:
        model = USM(
            dim=80,
            heads=4,
            ff_dim=256,
            depth=depth,
            depthwise_conv_kernel_size=31,
            dropout=0.1,
            use_group_norm=True,
            conv_first=True
        )
        input = torch.randn(10, 80, 100)
        lengths = torch.randint(10, 100, (10,))
        output, _ = model(input, lengths)
        assert output.shape == (10, 80, 100)

# Add more test cases...

# You can continue adding more test cases as needed.

