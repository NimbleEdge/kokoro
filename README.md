# Batch Implementation of Kokoro

[![License: Apache](https://img.shields.io/badge/License-Apache-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Kokoro is a text-to-speech model that generates high-quality speech from text input. This implementation supports batched inference and can be exported to ONNX for optimized deployment.


## Structure

```
kokoro/
├── __init__.py
├── model.py
├── tokenizer.py
│   ├── set_lexicon
│   └── phonemize
├── main.py
├── misaki_lexicons/
    ├── us_gold.json
    └── us_silver.json
```
Misaki Lexicons are taken from [hexgrad/misaki](https://github.com/hexgrad/misaki)

## Features

- High-quality text-to-speech synthesis
- Support for [batched inference](kokoro/model.py)
- Simplified [Tokenizer](kokoro/tokenizer.py) for On-device deployment
- ONNX export capability for optimized deployment
- Multiple [voice options](voices)


## Requirements
- Python >=3.10, <3.13
- huggingface_hub
- loguru
- numpy
- torch
- transformers
- ONNX Runtime >= 1.20.0 (for ONNX inference)

## Quick Start Linux

Here's a simple example to get started with Batch Kokoro:

```python
from kokoro import KModel, phonemize, set_lexicon
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


set_lexicon(json.load(open("./misaki_lexicons/us_gold.json")) | json.load(open("./misaki_lexicons/us_silver.json")))

# Create batch of input ids
input_id_tensors = []

for t in text:
    ps = phonemize(t)["ps"]
    toks = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), ps)))
    input_id_tensors.append(torch.tensor([0,*toks,0], dtype=torch.long))

input_lengths = torch.tensor([toks.shape[0] for toks in input_id_tensors], dtype=torch.long)
input_ids = rnn.pad_sequence(input_id_tensors, batch_first=True, padding_value=0)

# Generate speech
audio, pred_dur = model.forward_with_tokens(input_ids, 1.0, input_lengths)
```

## Example Output

Here's a [sample of Kokoro's Batch Inference](./output.wav):

<audio controls>
  <source src="output.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

```python
from scipy.io.wavfile import write
write("./output.wav", 24000, audio[batch_idx][0].numpy())
```

## Quick Start On-Device

### Using NimbleEdge Platform

Kokoro can be deployed on-device using the NimbleEdge platform, which provides optimized inference for mobile and edge devices.

#### Setting up NimbleEdge

1. Install the NimbleEdge SDK for your platform (iOS/Android)
2. Upload the ONNX model assets on the NimbleEdge portal


#### Key Features of on_device_workflow

- **Text-to-Speech**: Convert text to high-quality speech using the `run_text_to_speech_model` function
- **Phoneme Generation**: Accurate phonemization with stress marking using the `phonemize` function
- **Custom Pronunciation**: Support for custom lexicons through the `init` function
- **Integration with LLM**: Built-in integration with LLM for conversational applications
- **Memory Efficient**: Optimized for mobile and edge devices with limited resources

#### Entry Points for Native Integration

The script provides several entry points for integration with Kotlin/iOS applications:

1. `run_text_to_speech_model`: Converts text to speech audio
2. `prompt_llm`: Sends user input to the LLM
3. `get_next_str`: Retrieves streaming responses from the LLM
4. `set_context`: Sets conversation history for contextual responses
5. `init`: Initializes the model with a custom lexicon

#### Kotlin Integration Examples

Here's how to call these functions from Kotlin:

```kotlin
import ai.nimbleedge.NimbleNet

class KokoroManager() {
    
    // Initialize the model with lexicon
    NimbleNet.runMethod(
        "init",
        inputs = hashMapOf(
            "lexicon" to NimbleNetTensor(
                shape = null, data = lexiconArray, datatype = DATATYPE.JSON
            )
        )
    )
    
    // Generate speech from text
    NimbleNet.runMethod(
        "run_text_to_speech_model",
        inputs = hashMapOf(
            "text" to NimbleNetTensor(
                data = input,
                shape = null,
                datatype = DATATYPE.STRING
            )
        )
    )
    
    // Send user query to LLM
    NimbleNet.runMethod(
        "prompt_llm",
        inputs = hashMapOf(
            "query" to NimbleNetTensor(input, DATATYPE.STRING, null),
            "is_voice_initiated" to NimbleNetTensor(
                if (isVoiceInitiated) 1 else 0,
                DATATYPE.INT32,
                null
            )
        ),
    )
    
    // Get next chunk of LLM response
    NimbleNet.runMethod("get_next_str", hashMapOf())
    
    // Set conversation history context
    NimbleNet.runMethod(
        "set_context", hashMapOf(
            "context" to NimbleNetTensor(
                data = historicalContext,
                datatype = DATATYPE.JSON_ARRAY,
                shape = intArrayOf(historicalContext.length())
            )
        )
    )
}
```

#### Swift (iOS) Integration Examples

Here's how to call these functions from Swift:

```swift
import NimbleNet

// Initialize the model with lexicon
NimbleNet.runMethod(
    "init",
    inputs: [
        "lexicon": NimbleNetTensor(
            data: lexiconJson,
            dataType: .json,
            shape: nil
        )
    ]
)

// Generate speech from text
let speechResult = NimbleNet.runMethod(
    "run_text_to_speech_model",
    inputs: [
        "text": NimbleNetTensor(
            data: text,
            dataType: .string,
            shape: nil
        )
    ]
)
let audioData = speechResult["audio"]!.data as! Data

// Send user query to LLM
NimbleNet.runMethod(
    "prompt_llm",
    inputs: [
        "query": NimbleNetTensor(
            data: query,
            dataType: .string,
            shape: nil
        ),
        "is_voice_initiated": NimbleNetTensor(
            data: isVoiceInitiated ? 1 : 0,
            dataType: .int32,
            shape: nil
        )
    ]
)

// Get next chunk of LLM response
let responseChunk = NimbleNet.runMethod(
    "get_next_str",
    inputs: [:]
)
let text = responseChunk["str"]!.data as! String
let isFinished = (responseChunk["finished"]?.data as? Bool) ?? false

// Set conversation history context
let messageArray = messages.map { message -> [String: String] in
    return [
        "type": message.type, // "user" or "assistant"
        "message": message.content
    ]
}

NimbleNet.runMethod(
    "set_context",
    inputs: [
        "context": NimbleNetTensor(
            data: messageArray,
            dataType: .jsonArray,
            shape: [messageArray.count]
        )
    ]
)
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


