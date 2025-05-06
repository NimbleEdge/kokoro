# Kokoro

[![License: Apache](https://img.shields.io/badge/License-Apache-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Kokoro is a text-to-speech model that generates high-quality speech from text input. This implementation supports batched inference and can be exported to ONNX for optimized deployment.

## Features

- High-quality text-to-speech synthesis
- Support for batched inference
- Simplified Tokenizer for On-device deployment
- ONNX export capability for optimized deployment
- Multiple voice options

## Requirements
- Python >=3.10, <3.13
- huggingface_hub
- loguru
- numpy
- torch
- transformers
- ONNX Runtime (for ONNX inference)

## Quick Start

Here's a simple example to get started with Batch Kokoro:

```python
from kokoro import KModel, phonemize
import torch
import torch.nn.utils.rnn as rnn

# Initialize model
text = [
    "This is a test!",
    "This is a test! I'm going to the store.",
]
model = KModel(
    repo_id="hexgrad/Kokoro-82M",
    disable_complex=True,
    voice_name="af_heart"
).to("cpu").eval()

# Create batch of input ids
input_id_tensors = []
for t in text:
    ps, mtoks = phonemize(t)
    toks = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), ps)))
    input_id_tensors.append(torch.tensor([0,*toks,0], dtype=torch.long))

input_lengths = torch.tensor([toks.shape[0] for toks in input_id_tensors], dtype=torch.long)
input_ids = rnn.pad_sequence(input_id_tensors, batch_first=True, padding_value=0)

# Generate speech
audio, pred_dur = model.forward_with_tokens(input_ids, 1.0, input_lengths)
```

## ONNX Export and Inference

### Export to ONNX

To export the model to ONNX format:

```bash
python export_onnx.py
```

### ONNX Inference

Here's how to use the exported ONNX model:

```python
import onnxruntime as ort
from scipy.io.wavfile import write

# Create inference session
session = ort.InferenceSession("./onnx_models/kokoro_batched_quantized.onnx")

# Run inference
input_ids = torch.randint(0, 100, (1, 100))
input_lengths = torch.randint(1, 100, (1,))
audio, pred_dur = session.run(
    None, 
    {
        "input_ids": input_ids, 
        "speed": 1.0, 
        "input_lengths": input_lengths
    }
)

# Save audio
write("output.wav", 24000, audio)
```

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


