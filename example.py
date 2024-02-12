import torch
from usm_torch import USMEncoder

# Initialize model
model = USMEncoder(
    dim=80,  # Dimension of the input
    heads=4,  # Number of attention heads
    ff_dim=128,  # Dimension of the feed-forward layer
    depth=4,  # Number of transformer layers
    depthwise_conv_kernel_size=31,  # Kernel size for depthwise convolution
    dropout=0.5,  # Dropout rate
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

# Forward pass
outputs, output_lengths = model(inputs, lengths)  # Perform forward pass
print(f"outputs.shape: {outputs.shape}")  # Print the shape of the output tensor
print(
    f"output_lengths.shape: {output_lengths.shape}"
)  # Print the shape of the output lengths tensor
