import torch
from usm import USM

# Initialize model
model = USM(
    dim=80,
    heads=4,
    ff_dim=128,
    depth=4,
    depthwise_conv_kernel_size=31,
    dropout=0.5,
)

# Example input
batch_size = 10
max_length = 400
lengths = torch.randint(1, max_length, (batch_size,))
inputs = torch.rand(batch_size, int(lengths.max()), 80)  # Assuming input_dim=80

# Forward pass
outputs, output_lengths = model(inputs, lengths)
print(f"outputs.shape: {outputs.shape}")
print(f"output_lengths.shape: {output_lengths.shape}")