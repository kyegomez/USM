[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# USM
Implementation of Google's universal speech model from the paper: [Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages](https://arxiv.org/pdf/2303.01037.pdf)
I'm implementing this mostly because Gemini the all-new multi-modality foundation model from google uses it! [Check out our Gemini implementation here:](https://github.com/kyegomez/Gemini)


# Install
`pip install usm-torch`


## Usage
```python
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
lengths = torch.randint(1, max_length, (batch_size,))  # Randomly generate sequence lengths
inputs = torch.rand(batch_size, int(lengths.max()), 80)  # Randomly generate input tensor

# Forward pass
outputs, output_lengths = model(inputs, lengths)  # Perform forward pass
print(f"outputs.shape: {outputs.shape}")  # Print the shape of the output tensor
print(f"output_lengths.shape: {output_lengths.shape}")  # Print the shape of the output lengths tensor


```

# License
MIT

# Citation
```bibtex
@misc{zhang2023google,
    title={Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages}, 
    author={Yu Zhang and Wei Han and James Qin and Yongqiang Wang and Ankur Bapna and Zhehuai Chen and Nanxin Chen and Bo Li and Vera Axelrod and Gary Wang and Zhong Meng and Ke Hu and Andrew Rosenberg and Rohit Prabhavalkar and Daniel S. Park and Parisa Haghani and Jason Riesa and Ginger Perng and Hagen Soltau and Trevor Strohman and Bhuvana Ramabhadran and Tara Sainath and Pedro Moreno and Chung-Cheng Chiu and Johan Schalkwyk and Françoise Beaufays and Yonghui Wu},
    year={2023},
    eprint={2303.01037},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

```


## Todo
- [ ] Implement the proj -> cosine similarity -> codebook
- [ ] Implement chunk wise attention
- [ ] Implement on paired input, with the text encoder: embed extractor -> resampler -> refiner -> text embedding, RNN-T reconstruction loss
- [ ] Text input: text input -> speech encoder -> text decoder -> rnn-t reconstruction
