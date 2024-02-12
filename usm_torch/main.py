
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchaudio.models.conformer import Conformer


def codebook(
    speech_features: torch.Tensor, num_quantization_targets: int, dim: int
) -> torch.Tensor:
    """
    This function creates a codebook for the BEST-RQ pre-training method.

    Args:
        speech_features (torch.Tensor): The 3D tensor of speech features to be quantized.
                                        The dimensions should be (batch_size, sequence_length, feature_dim).
        num_quantization_targets (int): The number of quantization targets.
        dim (int): The dimension of the speech features and the codebook vectors.

    Returns:
        labels (torch.Tensor): The labels of the speech features, obtained by finding the closest codebook vector for each feature.
    """

    # Check if the input is a 3D tensor (batch_size, sequence_length, feature_dim)
    if len(speech_features.shape) != 3:
        raise ValueError(
            "The input speech features should be a 3D tensor with dimensions (batch_size, sequence_length, feature_dim)."
        )

    # Get the dimensions of the speech features
    batch_size, sequence_length, feature_dim = speech_features.size()

    # Initialize the codebook vectors randomly
    codebook_vectors = torch.randn(num_quantization_targets, dim)

    # Initialize the projection matrix randomly and freeze it
    projection_matrix = nn.Linear(dim, dim)
    for param in projection_matrix.parameters():
        param.requires_grad = False

    # Project the speech features into the embedding space
    projected_features = projection_matrix(speech_features.view(-1, dim)).view(
        batch_size, sequence_length, dim
    )

    # Initialize a tensor to store the labels of the speech features
    labels = torch.zeros(batch_size, sequence_length)

    # For each projected feature, find the closest codebook vector
    for i in range(batch_size):
        for j in range(sequence_length):
            # Calculate the cosine similarity between the projected feature and each codebook vector
            similarities = F.cosine_similarity(
                projected_features[i, j].unsqueeze(0), codebook_vectors
            )

            # Find the index of the codebook vector with the highest similarity
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
    # Project the input to the codebook dimension
    x = nn.Linear(dim, dim)(x)

    # Cosine similarity
    x = nn.CosineSimilarity(dim=1, eps=1e-6)(x, x)

    # Codebook
    return codebook(x, num_codebooks, dim)


x = torch.randn(10, 100, 100)
print(audio_to_codebook(x, 100, 100))


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
