import torch
import numpy as np
import os
import sys
import inspect
import traceback

# Import both implementations
import kokoro as batched_kokoro
import kokoro_orig as orig_kokoro

# Remove the fix application
# import fix_batched_kokoro
# fix_batched_kokoro.apply_fixes()

print("Starting test_batch.py...")

# Define a simple compare_tensors for backward compatibility
def compare_tensors(t1, t2, name, tolerance=1e-5):
    return compare_tensors_with_adaptation(t1, t2, name, tolerance)

def adapt_tensor_dims(tensor, reference_tensor, tensor_name=None):
    """
    Adapt a tensor's dimensions to match the reference tensor shape
    without changing the actual data.
    """
    if tensor is None or reference_tensor is None:
        return tensor
    
    if tensor.shape == reference_tensor.shape:
        return tensor
    
    adapted = tensor
    
    # Special case for text_encoder_output - needs transposition
    if tensor_name == "text_encoder_final_output":
        if tensor.dim() >= 3 and reference_tensor.dim() >= 3:
            if tensor.shape[0] == reference_tensor.shape[0]:
                adapted = tensor.transpose(1, 2)
                return adapted
    
    # Special case for decoder_output - original has extra dimension
    if tensor_name == "decoder_output":
        if reference_tensor.dim() == 3 and tensor.dim() == 2:
            if reference_tensor.shape[0] == tensor.shape[0] and reference_tensor.shape[2] == tensor.shape[1]:
                adapted = tensor.unsqueeze(1)
                print(f"Adapted decoder_output: {tensor.shape} -> {adapted.shape}")
                return adapted
    
    # Case 1: Extra batch dimension (e.g., [1, 1, 56] vs [1, 56])
    if tensor.dim() == reference_tensor.dim() + 1 and tensor.size(1) == 1:
        adapted = tensor.squeeze(1)
    
    # Case 2: Different ordering of dimensions (e.g., [1, 20, 512] vs [1, 512, 20])
    elif tensor.dim() == reference_tensor.dim() and tensor.dim() >= 3:
        # Check if dimensions match when swapped
        if tensor.shape[0] == reference_tensor.shape[0]:  # First dimension usually batch size
            # Check if dimensions 1 and 2 are swapped
            if tensor.shape[1] == reference_tensor.shape[2] and tensor.shape[2] == reference_tensor.shape[1]:
                adapted = tensor.transpose(1, 2)
    
    # Case 3: Different number of dimensions with more complex pattern
    elif tensor.dim() != reference_tensor.dim():
        # General case for adding a dimension in the middle when reference has 3 dims and tensor has 2
        if reference_tensor.dim() == 3 and tensor.dim() == 2:
            if reference_tensor.shape[0] == tensor.shape[0]:
                adapted = tensor.unsqueeze(1)
        # Handle specific cases based on the shapes we've seen
        # For example: F0Ntrain outputs [1, 1, 56] vs [1, 56]
        elif tensor.dim() == 3 and tensor.size(1) == 1 and reference_tensor.dim() == 2:
            adapted = tensor.squeeze(1)
    
    return adapted

def compare_tensors_with_adaptation(t1, t2, name, tolerance=1e-5):
    """Compare two tensors with dimension adaptation and report differences"""
    # Basic checks for None
    if t1 is None or t2 is None:
        print(f"[DIFF] {name}: One is None, other is {t1 if t1 is not None else t2}")
        return False
    
    # 1. FIRST - Special case for LSTM outputs which return [PackedSequence, (h_n, c_n)]
    if name.endswith("lstm_output"):
        print(f"[INFO] {name}: Processing LSTM output")
        # Check if we have two elements regardless of container type
        if hasattr(t1, "__len__") and len(t1) == 2 and hasattr(t2, "__len__") and len(t2) == 2:
            # First element should be PackedSequence
            if hasattr(t1[0], 'data') and hasattr(t2[0], 'data'):
                print(f"[INFO] {name}: Comparing PackedSequence output data")
                result1 = compare_tensors_with_adaptation(t1[0].data, t2[0].data, f"{name}_output", tolerance)
                
                # Second element should be tuple of hidden states (h_n, c_n)
                print(f"[INFO] {name}: Comparing hidden states")
                result2 = True
                if hasattr(t1[1], "__len__") and hasattr(t2[1], "__len__") and len(t1[1]) == len(t2[1]):
                    for i, (h1, h2) in enumerate(zip(t1[1], t2[1])):
                        sub_result = compare_tensors_with_adaptation(h1, h2, f"{name}_hidden_{i}", tolerance)
                        result2 = result2 and sub_result
                else:
                    print(f"[DIFF] {name}: Hidden state structure mismatch")
                    result2 = False
                
                return result1 and result2
    
    # 2. THEN - Handle PackedSequence
    if hasattr(torch.nn.utils.rnn, 'PackedSequence'):
        if isinstance(t1, torch.nn.utils.rnn.PackedSequence) and isinstance(t2, torch.nn.utils.rnn.PackedSequence):
            print(f"[INFO] {name}: Comparing PackedSequence data")
            return compare_tensors_with_adaptation(t1.data, t2.data, f"{name}_data", tolerance)
        elif isinstance(t1, torch.nn.utils.rnn.PackedSequence):
            print(f"[INFO] {name}: First is PackedSequence, second is not. Comparing data to tensor")
            return compare_tensors_with_adaptation(t1.data, t2, f"{name}_data", tolerance)
        elif isinstance(t2, torch.nn.utils.rnn.PackedSequence):
            print(f"[INFO] {name}: Second is PackedSequence, first is not. Comparing tensor to data")
            return compare_tensors_with_adaptation(t1, t2.data, f"{name}_data", tolerance)
    
    # 3. THEN - General tuple handling
    if isinstance(t1, (list, tuple)) and isinstance(t2, (list, tuple)):
        # Handle different types of containers
        if len(t1) != len(t2):
            print(f"[DIFF] {name}: Container length mismatch {len(t1)} vs {len(t2)}")
            return False
        
        # Compare each item in the list/tuple
        print(f"[INFO] {name}: Comparing container with {len(t1)} items")
        result = True
        for i in range(len(t1)):
            item_result = compare_tensors_with_adaptation(t1[i], t2[i], f"{name}[{i}]", tolerance)
            result = result and item_result
        return result
    
    # For tensors, adapt dimensions before comparison
    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
        # Adapt t2 to match t1's dimensions
        adapted_t2 = adapt_tensor_dims(t2, t1, name)
        
        if t1.shape != adapted_t2.shape:
            print(f"[DIFF] {name}: Shape mismatch after adaptation {t1.shape} vs {adapted_t2.shape}")
            return False
        
        # Always print some sample values for debugging
        print(f"[COMPARE] {name}: Shape: {t1.shape}")
        
        if t1.numel() > 0:
            # Sample some values (first, middle, last)
            sample_idx = [0, t1.numel() // 2, min(t1.numel() - 1, 5)]
            flat_t1 = t1.reshape(-1)
            flat_t2 = adapted_t2.reshape(-1)
            
            print(f"[SAMPLE] {name}: First few values comparison:")
            for idx in sample_idx:
                orig_val = flat_t1[idx].item()
                batch_val = flat_t2[idx].item()
                diff = abs(orig_val - batch_val)
                print(f"  Index {idx}: Original={orig_val:.6f}, Batched={batch_val:.6f}, Diff={diff:.6f}")
        
        # Compare values
        diff = torch.abs(t1 - adapted_t2)
        max_diff = torch.max(diff).item()
        print(f"[INFO] {name}: Max difference: {max_diff:.6f}")
        
        # Calculate mean absolute difference
        mean_diff = torch.mean(diff).item()
        print(f"[INFO] {name}: Mean difference: {mean_diff:.6f}")
        
        # Calculate percentage of elements with significant differences
        sig_diff_count = torch.sum(diff > tolerance).item()
        sig_diff_percentage = (sig_diff_count / t1.numel()) * 100
        print(f"[INFO] {name}: {sig_diff_count} elements ({sig_diff_percentage:.2f}%) exceed tolerance {tolerance}")
        
        if max_diff > tolerance:
            print(f"[DIFF] {name}: Values differ (max diff: {max_diff:.6f})")
            
            # Find indices of largest differences
            flat_diff = diff.reshape(-1)
            _, top_indices = torch.topk(flat_diff, min(3, flat_diff.numel()))
            print(f"[DIFF] {name}: Top 3 largest differences:")
            for idx in top_indices:
                idx = idx.item()
                orig_val = flat_t1[idx].item()
                batch_val = flat_t2[idx].item()
                diff_val = flat_diff[idx].item()
                print(f"  Index {idx}: Original={orig_val:.6f}, Batched={batch_val:.6f}, Diff={diff_val:.6f}")
            
            return False
        
        print(f"[MATCH] {name}: Values match within tolerance {tolerance}")
        return True
    
    # For other types, just compare directly
    equal = t1 == t2
    print(f"[COMPARE] {name}: {t1} == {t2} -> {equal}")
    return equal

# Create custom hooks to capture tensor values at specific points
class TracingHook:
    def __init__(self, name, output_dict):
        self.name = name
        self.output_dict = output_dict
        
    def __call__(self, module, inputs, outputs):
        # Store outputs as before
        self.output_dict[self.name] = outputs
        if isinstance(outputs, tuple) and len(outputs) > 0:
            self.output_dict[f"{self.name}_item0"] = outputs[0]
            if len(outputs) > 1:
                self.output_dict[f"{self.name}_item1"] = outputs[1]
        # Log output shapes as before
        print(f"[HOOK] Captured {self.name} with shape: {outputs.shape if isinstance(outputs, torch.Tensor) else [o.shape if isinstance(o, torch.Tensor) else type(o) for o in outputs]}")

# Create a hook for capturing method outputs
def add_method_hooks(model, method_name, hook_name, output_dict):
    """Add a hook to a method to capture its output"""
    original_method = getattr(model, method_name)
    
    def hook_wrapper(*args, **kwargs):
        # Capture input arguments
        if method_name == "reshape_for_batch":
            # Check if we have the right number of arguments
            if len(args) >= 3:
                output_dict[f"{hook_name}_input_x"] = args[1]
                output_dict[f"{hook_name}_input_lengths"] = args[2]
                print(f"[METHOD HOOK] {hook_name} inputs: x.shape={args[1].shape}, lengths={args[2]}")
        
        result = original_method(*args, **kwargs)
        output_dict[hook_name] = result
        if isinstance(result, torch.Tensor):
            print(f"[METHOD HOOK] {hook_name}: shape={result.shape}")
        elif isinstance(result, tuple) and len(result) > 0:
            print(f"[METHOD HOOK] {hook_name}: tuple with shapes={[t.shape if isinstance(t, torch.Tensor) else type(t) for t in result]}")
        return result
    
    setattr(model, method_name, hook_wrapper)
    return hook_wrapper

def run_comparison(phonemes, ref_s):
    """Run both implementations and compare outputs at each step"""
    # Create models
    orig_model = orig_kokoro.KModel().eval()
    batched_model = batched_kokoro.KModel().eval()
    
    # Step 1: Compare tokenization
    input_ids_orig = list(filter(lambda i: i is not None, map(lambda p: orig_model.vocab.get(p), phonemes)))
    input_ids_batched = list(filter(lambda i: i is not None, map(lambda p: batched_model.vocab.get(p), phonemes)))
    
    print("Comparing tokenization...")
    if input_ids_orig != input_ids_batched:
        print("[DIFF] Tokenization differs")
        return
    
    # Convert to tensors
    input_ids_orig = torch.LongTensor([[0, *input_ids_orig, 0]]).to(orig_model.device)
    input_ids_batched = torch.LongTensor([[0, *input_ids_batched, 0]]).to(batched_model.device)
    ref_s_orig = ref_s[0,:,:].to(orig_model.device)
    ref_s_batched = ref_s.to(batched_model.device)
    
    print(f"Input shapes - Original: {input_ids_orig.shape}, Batched: {input_ids_batched.shape}")
    print(f"Reference shapes - Original: {ref_s_orig.shape}, Batched: {ref_s_batched.shape}")
    
    # Run each step manually and compare
    print("Running forward passes and comparing intermediate tensors...")
    
    with torch.no_grad():
        # Create output dictionaries to store tensor values at various points
        orig_outputs = {}
        batched_outputs = {}
        
        # Add hooks to track reshape_for_batch in the batched model
        add_method_hooks(batched_model, "reshape_for_batch", "reshape_for_batch", batched_outputs)
        
        # Add hooks to track key modules in original model
        orig_model.bert.register_forward_hook(TracingHook("bert_output", orig_outputs))
        orig_model.bert_encoder.register_forward_hook(TracingHook("bert_encoder_output", orig_outputs))
        orig_model.predictor.text_encoder.register_forward_hook(TracingHook("predictor.text_encoder_output", orig_outputs))
        orig_model.predictor.lstm.register_forward_hook(TracingHook("predictor.lstm_output", orig_outputs))
        orig_model.predictor.shared.register_forward_hook(TracingHook("predictor.shared_output", orig_outputs))
        orig_model.predictor.duration_proj.register_forward_hook(TracingHook("predictor.duration_proj_output", orig_outputs))
        # Hook F0Ntrain function with method hook instead of forward hook
        add_method_hooks(orig_model.predictor, "F0Ntrain", "predictor.F0Ntrain_output", orig_outputs)
        orig_model.text_encoder.register_forward_hook(TracingHook("text_encoder_final_output", orig_outputs))
        orig_model.decoder.register_forward_hook(TracingHook("decoder_output", orig_outputs))
        orig_model.decoder.generator.m_source.l_sin_gen.register_forward_hook(TracingHook("generator.l_sin_gen_output", orig_outputs))
        # For original model - add hooks for decoder components
        orig_model.decoder.F0_conv.register_forward_hook(TracingHook("decoder.F0_conv_output", orig_outputs))
        orig_model.decoder.N_conv.register_forward_hook(TracingHook("decoder.N_conv_output", orig_outputs))
        orig_model.decoder.generator.register_forward_hook(TracingHook("decoder.generator_output", orig_outputs))
        # Add hooks to track key modules in batched model
        batched_model.bert.register_forward_hook(TracingHook("bert_output", batched_outputs))
        batched_model.bert_encoder.register_forward_hook(TracingHook("bert_encoder_output", batched_outputs))
        batched_model.predictor.text_encoder.register_forward_hook(TracingHook("predictor.text_encoder_output", batched_outputs))
        batched_model.predictor.lstm.register_forward_hook(TracingHook("predictor.lstm_output", batched_outputs))
        batched_model.predictor.shared.register_forward_hook(TracingHook("predictor.shared_output", batched_outputs))
        batched_model.predictor.duration_proj.register_forward_hook(TracingHook("predictor.duration_proj_output", batched_outputs))
        # Hook F0Ntrain function with method hook instead of forward hook
        add_method_hooks(batched_model.predictor, "F0Ntrain", "predictor.F0Ntrain_output", batched_outputs)
        batched_model.text_encoder.register_forward_hook(TracingHook("text_encoder_final_output", batched_outputs))
        batched_model.decoder.register_forward_hook(TracingHook("decoder_output", batched_outputs))
        batched_model.decoder.generator.m_source.l_sin_gen.register_forward_hook(TracingHook("generator.l_sin_gen_output", batched_outputs))
        # For batched model - add hooks for decoder components
        batched_model.decoder.F0_conv.register_forward_hook(TracingHook("decoder.F0_conv_output", batched_outputs))
        batched_model.decoder.N_conv.register_forward_hook(TracingHook("decoder.N_conv_output", batched_outputs))
        batched_model.decoder.generator.register_forward_hook(TracingHook("decoder.generator_output", batched_outputs))
        # Add hooks for generator's key components
        for orig_model_hook in [
            (orig_model.decoder.generator.f0_upsamp, "generator.f0_upsamp_output"),
            (orig_model.decoder.generator.m_source, "generator.m_source_output")
        ]:
            orig_model_hook[0].register_forward_hook(TracingHook(orig_model_hook[1], orig_outputs))

        for batched_model_hook in [
            (batched_model.decoder.generator.f0_upsamp, "generator.f0_upsamp_output"),
            (batched_model.decoder.generator.m_source, "generator.m_source_output")
        ]:
            batched_model_hook[0].register_forward_hook(TracingHook(batched_model_hook[1], batched_outputs))
        # Run forward pass
        print("\n--- Running original model ---")
        input_lengths_orig = torch.tensor([input_ids_orig.shape[1]], dtype=torch.long, device=input_ids_orig.device)
        audio_orig, dur_orig = orig_model.forward_with_tokens(input_ids_orig, ref_s_orig, 1.0)
        
        print("\n--- Running batched model ---")
        input_lengths_batched = torch.tensor([input_ids_batched.shape[1]], dtype=torch.long, device=input_ids_batched.device)
        audio_batched, dur_batched = batched_model.forward_with_tokens(
            input_ids_batched, ref_s_batched, 1.0, 
            input_lengths_batched)
        
        # Handle audio dimension difference - original model gives [1, 1, T] but batched model gives [1, T]
        if audio_orig.dim() == 3 and audio_batched.dim() == 2:
            # If batch model output has one fewer dimension, add it
            audio_batched = audio_batched.unsqueeze(1)
        elif audio_orig.dim() == 2 and audio_batched.dim() == 3:
            # If original model output has one fewer dimension, adapt batched
            audio_batched = audio_batched.squeeze(1)
        
        # Compare final outputs
        print("\n--- Comparing final audio outputs ---")
        print(f"Original audio shape: {audio_orig.shape}, Batched audio shape: {audio_batched.shape}")
        
        # For single-batch case, take the first item from batched output if needed
        if len(audio_batched) == 1:
            if len(audio_orig.shape) == 1:  # Handle scalar case
                audio_orig = audio_orig.unsqueeze(0)
                
            print("Comparing single batch audio outputs...")
            print(f"Audio shapes to compare: {audio_orig.shape} vs {audio_batched[0].shape}")
            
            # Directly use our comparison function to get detailed logs
            audio_match = compare_tensors_with_adaptation(audio_orig, audio_batched[0], "final_audio")
            
            if audio_match:
                print("\n--- RESULT: Audio outputs match within tolerance! ---")
            else:
                print("\n--- RESULT: Audio outputs differ, comparing intermediate outputs ---")
                # Compare key intermediate values
                comparison_points = [
                    "bert_output", "bert_encoder_output", "predictor.text_encoder_output", 
                    "predictor.lstm_output", "predictor.duration_proj_output", "predictor.F0Ntrain_output", 
                    "predictor.shared_output", "text_encoder_final_output", 
                    "decoder.F0_conv_output", "decoder.N_conv_output",
                    "generator.f0_upsamp_output", "generator.l_sin_gen_output", "generator.m_source_output",
                    "decoder.generator_output", "decoder_output"
                ]
                
                # Then compare module outputs as before
                print("\n--- Comparing module outputs ---")
                for point in comparison_points:
                    if point in orig_outputs and point in batched_outputs:
                        print(f"\nComparing {point}:")
                        orig_output = orig_outputs[point]
                        batched_output = batched_outputs[point]
                        
                        if isinstance(orig_output, tuple) and isinstance(batched_output, tuple):
                            for i in range(min(len(orig_output), len(batched_output))):
                                if isinstance(orig_output[i], torch.Tensor) and isinstance(batched_output[i], torch.Tensor):
                                    print(f"  Item {i} - Original shape: {orig_output[i].shape}, Batched shape: {batched_output[i].shape}")
                                    # Apply dimension adaptation for tuple items
                                    adapted_output = adapt_tensor_dims(batched_output[i], orig_output[i], point)
                                    print(f"  Item {i} - Adapted shape: {adapted_output.shape}")
                                    compare_tensors_with_adaptation(orig_output[i], batched_output[i], f"{point}[{i}]")
                                elif isinstance(orig_output[i], torch.Tensor) and isinstance(batched_output[i], torch.nn.utils.rnn.PackedSequence):
                                    print(f"  Item {i} - Original shape: {orig_output[i].shape}, Batched shape: {batched_output[i].data.shape}")
                                    compare_tensors_with_adaptation(orig_output[i], batched_output[i].data.unsqueeze(0), f"{point}[{i}]")
                                elif isinstance(orig_output[i], torch.nn.utils.rnn.PackedSequence) and isinstance(batched_output[i], torch.nn.utils.rnn.PackedSequence):
                                    print(f"  Item {i} - Original shape: {orig_output[i].data.shape}, Batched shape: {batched_output[i].data.shape}")
                                    compare_tensors_with_adaptation(orig_output[i].data, batched_output[i].data, f"{point}[{i}]")
                        else:
                            if isinstance(orig_output, torch.Tensor) and isinstance(batched_output, torch.Tensor):
                                print(f"  Original shape: {orig_output.shape}, Batched shape: {batched_output.shape}")
                                # Apply dimension adaptation
                                adapted_output = adapt_tensor_dims(batched_output, orig_output, point)
                                print(f"  Adapted shape: {adapted_output.shape}")
                            compare_tensors_with_adaptation(orig_output, batched_output, point)
                    else:
                        print(f"\nCouldn't compare {point}: missing in one of the outputs")
            
            print("\n--- Checking reshape_for_batch outputs ---")
            if "reshape_for_batch" in batched_outputs:
                print(f"reshape_for_batch output shape: {batched_outputs['reshape_for_batch'].shape}")
            
            # Check all invocations of reshape_for_batch
            reshape_keys = [k for k in batched_outputs.keys() if k.startswith("reshape_for_batch")]
            reshape_input_keys = [k for k in reshape_keys if "_input" in k]
            reshape_output_keys = [k for k in reshape_keys if not "_input" in k]
            
            if len(reshape_output_keys) > 1:
                print("\n--- Multiple reshape_for_batch invocations detected ---")
                for i, k in enumerate(reshape_output_keys):
                    print(f"\nInvocation {i+1}:")
                    input_x_key = f"{k}_input_x"
                    input_lengths_key = f"{k}_input_lengths"
                    
                    if input_x_key in batched_outputs and input_lengths_key in batched_outputs:
                        input_x = batched_outputs[input_x_key]
                        input_lengths = batched_outputs[input_lengths_key]
                        output = batched_outputs[k]
                        
                        print(f"  Input x shape: {input_x.shape}")
                        print(f"  Input lengths: {input_lengths}")
                        print(f"  Output shape: {output.shape}")
                        
                        # Verify the reshape operation manually
                        max_len = input_lengths.max()
                        end_indices = input_lengths.cumsum(dim=0)
                        start_indices = torch.cat([torch.zeros(1, device=end_indices.device, dtype=end_indices.dtype), 
                                              end_indices[:-1]])
                        
                        print(f"  Calculated max_len: {max_len}")
                        print(f"  Calculated end_indices: {end_indices}")
                        print(f"  Calculated start_indices: {start_indices}")
                        
                        # Check a few values in the output to see if they match expected values
                        if input_x.numel() > 0 and output.numel() > 0:
                            # Check the first few values in each batch
                            for b in range(min(2, input_lengths.size(0))):
                                start_idx = start_indices[b].item()
                                print(f"  Batch {b} - First few values from input_x[{start_idx}:] vs output[{b},0,:]")
                                print(f"    Input: {input_x[start_idx:start_idx+3, :3]}")
                                print(f"    Output: {output[b, 0:min(3, output.shape[1]), :3]}")
            
            # Check if the reshape_for_batch implementation is correct for both models
            print("\n--- Comparing reshape_for_batch implementations ---")
            print("Original implementation doesn't have reshape_for_batch method")
            print("Batched implementation reshape_for_batch method:")
            if hasattr(batched_model, "reshape_for_batch"):
                print(inspect.getsource(batched_model.__class__.reshape_for_batch))

if __name__ == "__main__":
    # Test with a simple example
    phonemes = "h eh l ou w er l d"
    torch.manual_seed(100)
    ref_s = torch.randn(1, 1, 256)  # Style reference tensor
    print("Starting comparison...")
    run_comparison(phonemes, ref_s)
    print("Comparison completed successfully.")