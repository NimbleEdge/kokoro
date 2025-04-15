# %%
from kokoro import KModel
import torch
from misaki import en
import numpy as np
import os
import torch.nn.utils.rnn as rnn
import onnx
from onnx import helper

def load_bin_voice(voice_path: str) -> torch.Tensor:
    """
    Load a .bin voice file as a PyTorch tensor.
    
    Args:
        voice_path: Path to the .bin voice file
        
    Returns:
        PyTorch tensor containing the voice data
    """
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    
    if not voice_path.endswith('.bin'):
        raise ValueError(f"Expected a .bin file, got: {voice_path}")
    
    # Load the binary file as a numpy array of float32 values
    voice_data = np.fromfile(voice_path, dtype=np.float32).reshape(-1, 1, 256)
    # Convert to PyTorch tensor
    voice_tensor = torch.tensor(voice_data, dtype=torch.float32)
    
    # Return the tensor
    return voice_tensor

g2p = en.G2P(trf=False, british=False, fallback=None, unk='')

# Example 1: Using the pipeline with a voice name
text = [
    "This is a test!",
    "This is a test! I'm going to the store.",
]
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).to("cpu").eval()
ref_s = load_bin_voice("kokoro.js/voices/af_heart.bin")
input_id_tensors = []
style_tensors = []

for t in text:
    ps, mtoks = g2p(t)
    toks = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), ps)))
    input_id_tensors.append(torch.tensor([0,*toks,0], dtype=torch.long))
    style_tensors.append(ref_s[len(toks)-1][None,:,:])
input_lengths = torch.tensor([toks.shape[0] for toks in input_id_tensors], dtype=torch.long)
style_tensor = torch.cat(style_tensors, dim=0)
input_ids = rnn.pad_sequence(input_id_tensors, batch_first=True, padding_value=0)
print(input_lengths, style_tensor.shape, input_ids.shape)

audio, pred_dur = model.forward_with_tokens(input_ids, style_tensor, 1.0,input_lengths)


import onnx
onnx_file = "./onnx_models/kokoro_batched.onnx"
model.forward = model.forward_with_tokens

speed = torch.tensor([1.0], dtype=torch.float32)

torch.onnx.export(
    model, 
    args = (input_ids, style_tensor, speed, input_lengths), 
    f = onnx_file, 
    export_params = True, 
    verbose = False, 
    input_names = [ 'input_ids', 'style', 'speed', 'input_lengths' ], 
    output_names = [ 'waveform'], # 'duration' ],
    opset_version = 20, 
    dynamic_axes = {
        'input_ids': { 0: 'batch_size', 1: 'input_ids_len' }, 
        'style': { 0: 'batch_size' },
        'waveform': { 0: 'batch_size', 1: 'num_samples' }, 
        # 'duration': { 0: 'batch_duration' },
        'input_lengths': { 0: 'batch_size' },
    }, 
    do_constant_folding = False, 
)

def remove_problematic_nodes(input_model_path, output_model_path):
    # Load the model
    model = onnx.load(input_model_path)
    
    # Nodes to be removed
    nodes_to_remove = [
        "/Gather_5",
        "/Equal_1",
        "/Expand_1", 
        "/Where_1",
        "/Shape_7",
        "/ConstantOfShape_1",
        "/SplitToSequence",
        "/SplitToSequence_1",
        "/SequenceAt"
    ]
    
    # Find outputs of nodes to be removed
    outputs_of_removed_nodes = set()
    for node in model.graph.node:
        if node.name in nodes_to_remove:
            for output_name in node.output:
                outputs_of_removed_nodes.add(output_name)
    
    # Determine dependent nodes (nodes that consume outputs of removed nodes)
    dependent_nodes = set()
    for node in model.graph.node:
        if node.name not in nodes_to_remove:
            for input_name in node.input:
                if input_name in outputs_of_removed_nodes:
                    dependent_nodes.add(node.name)
                    break
    
    # Add dependent nodes to the removal list
    all_nodes_to_remove = set(nodes_to_remove).union(dependent_nodes)
    print(f"Original nodes to remove: {len(nodes_to_remove)}")
    print(f"Additional dependent nodes to remove: {len(dependent_nodes)}")
    print(f"Total nodes to remove: {len(all_nodes_to_remove)}")
    
    # Keep track of all outputs from removed nodes to clean up connections
    all_removed_inputs = set()
    for node in model.graph.node:
        if node.name in all_nodes_to_remove:
            for input_name in node.input:
                all_removed_inputs.add(input_name)

    # Create a new graph without the problematic nodes
    new_nodes = []
    for node in model.graph.node:
        if node.name not in nodes_to_remove:
            # Remove references to inputs of deleted nodes
            new_outputs = [output_name for output_name in node.output 
                          if output_name not in all_removed_inputs]
            
            # Only keep the node if it still has all its required inputs
            if new_outputs:
                # Create a new node with updated inputs
                new_node = helper.make_node(
                    op_type=node.op_type,
                    inputs=node.input,
                    outputs=new_outputs,
                    name=node.name
                )
                # Copy attributes
                for attr in node.attribute:
                    new_node.attribute.append(attr)
                new_nodes.append(new_node)
            else:
                new_nodes.append(node)
    
    # Create a new graph without the removed nodes
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer
    )
    
    # Create a new model with the modified graph
    new_model = helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string
    )
    
    # Validate and save the modified model
    onnx.checker.check_model(new_model)
    onnx.save(new_model, output_model_path)
    print(f"Saved cleaned model to {output_model_path}")
    
    return new_model

# # Fix random ops first
remove_problematic_nodes(onnx_file, onnx_file)
# print('Fixed random operations in kokoro.onnx')

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
onnx.checker.check_model(onnx.load(model_fp32))
print('onnx preprocess check ok!')

model_quant = "./onnx_models/kokoro_batched_quantized.onnx"
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QInt8)

# Then remove dangling nodes
# remove_dangling_nodes(model_quant, model_quant)
# print('Removed dangling nodes from kokoro.onnx')

onnx.checker.check_model(onnx.load(model_quant))
print('onnx quantized check ok!')

