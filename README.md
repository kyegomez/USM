[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# USM
Implementation of Google's universal speech model from the paper: [Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages](https://arxiv.org/pdf/2303.01037.pdf)
I'm implementing this mostly because Gemini the all-new multi-modality foundation model from google uses it! [Check out our Gemini implementation here:](https://github.com/kyegomez/Gemini)


# Install
`pip install usm-torch`


## Usage
```python
import torch
from usm_torch import USM

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


