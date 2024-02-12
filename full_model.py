import torch
from usm_torch.main import USM

# Create an instance of the USM model with specified parameters
usm_model = USM(
    dim=80,  # Dimension of the input data
    heads=4,  # Number of attention heads
    ff_dim=256,  # Dimension of the feed-forward layer
    depth=4,  # Number of transformer layers
    depthwise_conv_kernel_size=32,  # Kernel size for depthwise convolution
    dropout=0.1,  # Dropout rate
    use_group_norm=True,  # Whether to use group normalization
    conv_first=True,  # Whether to apply convolution before self-attention
    codebook=True,  # Whether to use codebook attention
)

# Example input
batch_size = 10  # Number of samples in a batch
max_length = 400  # Maximum length of the input sequence
lengths = torch.randint(
    1, max_length, (batch_size,)
)  # Randomly generate sequence lengths
inputs = torch.rand(
    batch_size, int(lengths.max()), 80
)  # Randomly generate input tensor

# Forward pass through the USM module
output = usm_model(inputs, lengths)

# Print the output shape
print(output.shape)
