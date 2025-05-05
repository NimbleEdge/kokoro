from kokoro import KModel
from kokoro_orig import KModel as KModel_orig
import torch
from misaki import en
import numpy as np
import os
import IPython.display as ipd
import torch.nn.utils.rnn as rnn
import time

torch.set_num_threads(3)
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


def run_benchmark(num_samples=5, num_iterations=10):
    print(f"Running benchmark with {num_samples} samples, {num_iterations} iterations each")
    print("-" * 50)
    
    # Initialize pipeline with American English
    g2p = en.G2P(trf=False, british=False, fallback=None, unk='')

    # Create sample texts
    text = ["How about trying something new?"] * num_samples
    
    # Load models
    print("Loading models...")
    model_orig = KModel_orig(repo_id="hexgrad/Kokoro-82M").to("cpu").eval()
    model = KModel(repo_id="hexgrad/Kokoro-82M").to("cpu").eval()
    
    # Load voice reference
    ref_s = load_bin_voice("kokoro.js/voices/af_heart.bin")
    
    # Process text inputs
    input_id_tensors = []
    input_lengths = []
    
    print("Preprocessing text...")
    for t in text:
        ps, mtoks = g2p(t)
        toks = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), ps)))
        input_id_tensors.append(torch.tensor([0, *toks, 0], dtype=torch.long))
        input_lengths.append(len(toks) + 2)  # +2 for start/end tokens
    
    input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.long)
    padded_input_ids = rnn.pad_sequence(input_id_tensors, batch_first=True, padding_value=0)
    
    print(f"Input shapes: lengths={input_lengths_tensor.shape}, padded_ids={padded_input_ids.shape}")
    
    # Benchmark original model (sequential processing)
    print("\nBenchmarking original model (sequential processing)...")
    sequential_times = []
    
    for iteration in range(num_iterations):
        start = time.time()
        
        # Process each sample individually (sequentially)
        for i in range(num_samples):
            # Use each sample individually
            sample_tensor = input_id_tensors[i][None, :]  # Add batch dimension
            audio, _ = model_orig.forward_with_tokens(
                sample_tensor, 
                ref_s[input_lengths[i]], 
                1.0
            )
            
        end = time.time()
        elapsed = end - start
        sequential_times.append(elapsed)
        print(f"  Iteration {iteration+1}: {elapsed:.4f}s")
    
    avg_sequential = sum(sequential_times) / len(sequential_times)
    print(f"\nOriginal model (sequential) - Average time: {avg_sequential:.4f}s")
    
    # Benchmark batched model
    print("\nBenchmarking batched model (parallel processing)...")
    batched_times = []
    
    for iteration in range(num_iterations):
        start = time.time()
        
        # Process all samples at once (batched)
        audio, frame_lengths = model.forward_with_tokens(
            padded_input_ids, 
            1.0,
            input_lengths_tensor
        )
        
        end = time.time()
        elapsed = end - start
        batched_times.append(elapsed)
        print(f"  Iteration {iteration+1}: {elapsed:.4f}s")
    
    avg_batched = sum(batched_times) / len(batched_times)
    print(f"\nBatched model - Average time: {avg_batched:.4f}s")
    
    # Calculate speedup
    speedup = avg_sequential / avg_batched
    print(f"\nResults summary:")
    print(f"  Original model (sequential): {avg_sequential:.4f}s")
    print(f"  Batched model (parallel): {avg_batched:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    run_benchmark(num_samples=3, num_iterations=20)