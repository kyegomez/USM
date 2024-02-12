import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchaudio.models.conformer import Conformer
from einops import rearrange


def codebook(
    x: Tensor,
    num_quant_targets: int,
    dim: int,
):
    if len(x.shape) != 3:
        raise ValueError(
            "The input speech features should be a 3D tensor with dimensions (batch_size, sequence_length, feature_dim)."
        )

    # Get the dimensions of the speech features
    b, s, d = x.size()

    # Init codebook vectors randomly
    codebook_vectors = torch.randn(num_quant_targets, dim)

    # Initialize the projection matrix randomly and freeze it
    projection_matrix = nn.Linear(dim, dim)
    for param in projection_matrix.parameters():
        param.requires_grad = False

    # Project the speech features into the embedding space
    projected_features = projection_matrix(x.view(-1, dim).view(b, s, d))

    # Initialize a tensor to store the labels of the speech features
    labels = torch.zeros(b, s)

    # For each projected feature, find the closest codebook vector
    for i in range(b):
        for j in range(s):
            # Calculate the cosine similarity between the projected feature and each codebook vector
            similarities = F.cosine_similarity(
                projected_features[i, j].unsqueeze(0), codebook_vectors
            )

            # Find the index of the codebook vectors with the highest similarity
            labels[i, j] = torch.argmax(similarities)

    return labels


def audio_to_codebook(
    x: Tensor,
    dim: int,
    num_codebooks: int,
):
    """
    Converts audio input to a codebook representation.

    Args:
        x (Tensor): The input tensor.
        dim (int): The dimension of the input tensor.
        num_codebooks (int): The number of codebooks to create.

    Returns:
        Tensor: The codebook representation of the audio input.
    """
    b, s, d = x.shape

    # Project the input to the codebook dimension
    x = nn.Linear(dim, dim)(x)

    # Cosine similarity
    x = nn.CosineSimilarity(dim=1, eps=1e-6)(x, x)
    print(x.shape)
    x = rearrange(x, "b s -> b s ()")
    b_b, b_s, b_d = x.shape
    x = nn.Linear(b_d, dim)(x)

    # Codebook
    return codebook(x, num_codebooks, dim)


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
                        *args,
                        **kwargs
                    ),
                    nn.Dropout(dropout),
                )
            )

        # Codebook

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


class USM(nn.Module):
    """
    USM (Universal Speech Model) module.

    Args:
        dim (int): Dimension of the model.
        heads (int): Number of attention heads.
        ff_dim (int): Dimension of the feed-forward layer.
        depth (int): Number of transformer layers.
        depthwise_conv_kernel_size (int): Kernel size for depthwise convolution.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        use_group_norm (bool, optional): Whether to use group normalization. Defaults to False.
        conv_first (bool, optional): Whether to apply convolution before self-attention. Defaults to False.

    Attributes:
        dim (int): Dimension of the model.
        heads (int): Number of attention heads.
        ff_dim (int): Dimension of the feed-forward layer.
        depth (int): Number of transformer layers.
        depthwise_conv_kernel_size (int): Kernel size for depthwise convolution.
        dropout (float): Dropout rate.
        use_group_norm (bool): Whether to use group normalization.
        conv_first (bool): Whether to apply convolution before self-attention.
        encoder (USMEncoder): USMEncoder instance.

    Examples:
    >>> from usm import USM
    >>> model = USM(
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
        dim: int,
        heads: int,
        ff_dim: int,
        depth: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        conv_first: bool = False,
        codebook: bool = False,
        *args,
        **kwargs
    ):
        super(USM, self).__init__()
        self.dim = dim
        self.heads = heads
        self.ff_dim = ff_dim
        self.depth = depth
        self.depthwise_conv_kernel_size = depthwise_conv_kernel_size
        self.dropout = dropout
        self.use_group_norm = use_group_norm
        self.conv_first = conv_first
        self.codebook = codebook

        self.encoder = USMEncoder(
            dim=dim,
            heads=heads,
            ff_dim=ff_dim,
            depth=depth,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=use_group_norm,
            conv_first=conv_first,
            *args,
            **kwargs
        )

    def forward(self, x, lengths) -> Tensor:
        """
        Forward pass of the USM module.

        Args:
            x (Tensor): Input tensor.
            lengths (Tensor): Lengths of the input sequences.

        Returns:
            Tensor: Output tensor.

        """
        encoded, _ = self.encoder(x, lengths)

        # Codebook
        if self.codebook:
            codebook = audio_to_codebook(encoded, self.dim, 10)
            print(codebook.shape)
            codebook = rearrange(codebook, "b s -> b s ()")
            codebook = nn.Linear(1, self.dim)(codebook)
            print(codebook.shape)

            concat = torch.cat((encoded, codebook), dim=1)

            return concat

        return encoded
