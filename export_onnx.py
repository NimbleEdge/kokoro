# %%
from kokoro import KModel, phonemize, set_lexicon
import torch
import torch.nn.utils.rnn as rnn
import onnx
import os
import shutil
import json

text = [
    "This is a test!",
    "This is a test! I'm going to the store.",
]
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True, voice_name="af_heart").to("cpu").eval()
set_lexicon(json.load(open("./misaki_lexicons/us_gold.json")) | json.load(open("./misaki_lexicons/us_silver.json")))

input_id_tensors = []

for t in text:
    ps = phonemize(t)["ps"]
    toks = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), ps)))
    input_id_tensors.append(torch.tensor([0,*toks,0], dtype=torch.long))
input_lengths = torch.tensor([toks.shape[0] for toks in input_id_tensors], dtype=torch.long)
input_ids = rnn.pad_sequence(input_id_tensors, batch_first=True, padding_value=0)
print(input_lengths, input_ids)

audio, pred_dur = model.forward_with_tokens(input_ids, 1.0,input_lengths)

if os.path.exists("./onnx_models/"):
    shutil.rmtree("./onnx_models/")
os.makedirs("./onnx_models/")

onnx_file = "./onnx_models/kokoro_batched.onnx"
model.forward = model.forward_with_tokens

speed = torch.tensor([1.0], dtype=torch.float16)
torch.onnx.export(
    model, 
    args = (input_ids, speed, input_lengths), 
    f = onnx_file, 
    export_params = True, 
    verbose = False, 
    input_names = [ 'input_ids', 'speed', 'input_lengths' ], 
    output_names = [ 'waveform', 'frame_lengths' ],
    opset_version = 20, 
    dynamic_axes = {
        'input_ids': { 0: 'batch_size', 1: 'input_ids_len' }, 
        'waveform': { 0: 'batch_size', 1: 'num_samples' }, 
        'frame_lengths': { 0: 'batch_size' },
        'input_lengths': { 0: 'batch_size' },
    }, 
    do_constant_folding = False, 
)

onnx.checker.check_model(onnx.load(onnx_file))
print('onnx check ok!')

from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
model_fp32 = "./onnx_models/kokoro_batched_preprocess.onnx"
quant_pre_process(
    input_model=onnx_file,
    output_model_path=model_fp32,
    skip_symbolic_shape=True,
    verbose=3,
)
model = onnx.load(model_fp32)
onnx.checker.check_model(model)
print('onnx preprocess check ok!')

model_quant = "./onnx_models/kokoro_batched_quantized.onnx"
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)

onnx.checker.check_model(onnx.load(model_quant))
print('onnx quantized check ok!')

