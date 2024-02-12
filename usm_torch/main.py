import torch
from torch import Tensor, nn
from torchaudio.models.conformer import Conformer


class USMEncoder(nn.Module):
    """Universal Speech Model (USMEncoder) is a model that can be used for any speech task.


    Args:
        dim (int): Dimension of the input features.
        heads (int): Number of heads in the multi-head attention.
        ff_dim (int): Dimension of the feed-forward
        depth (int): Number of layers in the model.
        depthwise_conv_kernel_size (int): Kernel size of the depthwise convolution.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        use_group_norm (bool, optional): Whether to use group normalization. Defaults to False.
        conv_first (bool, optional): Whether to use convolution first. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.


    Examples:
    >>> from usm import USMEncoder
    >>> model = USMEncoder(
    ...     dim=80,
    ...     heads=4,
    ...     ff_dim=256,
    ...     depth=4,
    ...     depthwise_conv_kernel_size=32,
    ...     dropout=0.1,
    ...     use_group_norm=True,
    ...     conv_first=True
    ... )
    >>> input = torch.randn(10, 80, 100)
    >>> output = model(input)


    """

    def __init__(
        self,
        dim,
        heads,
        ff_dim,
        depth,
        depthwise_conv_kernel_size,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        conv_first: bool = False,
        *args,
        **kwargs
    ):
        super(USMEncoder, self).__init__()
        self.dim = dim
        self.heads = heads
        self.ff_dim = ff_dim
        self.depth = depth
        self.depthwise_conv_kernel_size = depthwise_conv_kernel_size
        self.dropout = dropout
        self.use_group_norm = use_group_norm
        self.conv_first = conv_first

        # Ensure depthwise_conv_kernel_size is odd
        if depthwise_conv_kernel_size % 2 == 0:
            depthwise_conv_kernel_size += 1

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(dim),
                    Conformer(
                        input_dim=dim,
                        num_heads=heads,
                        ffn_dim=ff_dim,
                        num_layers=depth,
                        depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                        use_group_norm=use_group_norm,
                        convolution_first=conv_first,
                    ),
                    nn.Dropout(dropout),
                )
            )

    def forward(self, x: torch.Tensor, lengths) -> Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): _description_
            lengths (_type_): _description_

        Returns:
            Tensor: _description_
        """
        for layer in self.layers:
            x = layer[0](x)
            x, lengths = layer[1](x, lengths)
            x = layer[2](x)
        return x, lengths
