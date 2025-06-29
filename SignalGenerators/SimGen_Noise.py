import os
import gc
import psutil
import random
import pandas as pd
import numpy as np
from Bio import SeqIO
from scipy.signal import resample
from datetime import datetime
import concurrent.futures
import matplotlib.pyplot as plt

def normalize_signal(signal):
    mean = 63.63331112435138 # can be changed if needed, depending on dataset
    std = 10.705338783244988 # can be changed if needed, depending on dataset
    std = std if std != 0 else 1
    normalized = (signal - mean) / std
    print("mean is", mean, "and std is", std)

    return normalized
    
def moving_6mer_Substrings(string):
    return [string[i:i+6] for i in range(len(string) - 5)]

def predict_DNA_6mer_5_3_with_sampling(template, lut, lambda_time, sampling_rate=5000, I_max=180):
    template = template[::-1]  # 3' to 5'
    kmers = moving_6mer_Substrings(template)
    N = len(kmers)

    sampled_signals = []
    sampled_times = []
    current_time = 0.0
    i = 0

    backstep_prob = 0.005 # probability of backstep (can be changed) 
    skip_prob = 0.01  # probability of forward skip (can be changed) 
    max_skip = 2 # maximum number of skips (can be changed) 
    max_backstep = 2 # maximum number of backsteps (can be changed) 

    noise_std_fraction = 0.005 * I_max
    drift_per_second = 0.01 * I_max

    prev_mean = None
    step_count = 0

    while i < N:
        kmer = kmers[i]
        step_time = lambda_time
        ns = int(step_time * sampling_rate)
        if ns == 0:
            i += 1
            continue

        params = lut[kmer]
        pre_mean = params['pre_mean'] * I_max
        pre_std = params['pre_std'] * I_max
        post_mean = params['post_mean'] * I_max
        post_std = params['post_std'] * I_max

        if prev_mean is not None:
            pre_mean = 0.8 * pre_mean + 0.2 * prev_mean
        prev_mean = pre_mean

        drift = drift_per_second * (current_time / 1.0) # baseline drift

        pre_signal = np.random.normal(pre_mean, pre_std, ns) + np.random.normal(0, noise_std_fraction, ns)
        pre_signal += drift

        post_signal = np.random.normal(post_mean, post_std, ns) + np.random.normal(0, noise_std_fraction, ns)
        post_signal += drift

        smoothing_window = int(0.1 * ns) or 1
        smooth_noise = np.convolve(np.random.normal(0, 0.01 * I_max, ns + smoothing_window), 
                                   np.ones(smoothing_window) / smoothing_window, mode='valid')[:ns] # 1/f-like noise
        pre_signal += smooth_noise
        post_signal += smooth_noise[::-1]

        t_pre = np.linspace(current_time, current_time + step_time, ns, endpoint=False)
        current_time += step_time
        t_post = np.linspace(current_time, current_time + step_time, ns, endpoint=False)
        current_time += step_time

        sampled_signals.extend(pre_signal)
        sampled_times.extend(t_pre)
        sampled_signals.extend(post_signal)
        sampled_times.extend(t_post)

        r = np.random.rand()
        if r < backstep_prob and i > 0:
            i -= np.random.randint(1, max_backstep + 1)
            i = max(0, i)
        elif r < backstep_prob + skip_prob and i < N - 2:
            i += np.random.randint(2, max_skip + 2)  
        else:
            i += 1

    return pd.DataFrame({
        "time": sampled_times,
        "current": sampled_signals
    })

def extract_random_sequences(reference_genome_file, num_sequences, seq_length):
    genome = "".join(str(record.seq) for record in SeqIO.parse(reference_genome_file, "fasta"))
    genome_len = len(genome)
    sequences = set()
    
    while len(sequences) < num_sequences:
        start = random.randint(0, genome_len - seq_length)
        seq = genome[start:start + seq_length]
        sequences.add(seq)
    
    return list(sequences)

def encode_dna_sequence(seq):
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4} # if reference genome includes lower case, those can be added here 
    return [mapping.get(base.upper(), 0) for base in seq]

def generate_random_signals(reference_genome_file, num_reads, seq_length):
    sequences = extract_random_sequences(reference_genome_file, num_reads, seq_length)
    test_size = max(1, int(0.05 * len(sequences)))
    test_indices = np.random.choice(len(sequences), size=test_size, replace=False)
    test_sequences = [sequences[i] for i in test_indices]
    train_sequences = [sequences[i] for i in range(len(sequences)) if i not in test_indices]
    return train_sequences, test_sequences

def chunk_signal_optimized(n_chunks, signal, chunksize, overlap):
    step = chunksize - overlap
    result = np.zeros((n_chunks, chunksize))
    
    for i in range(n_chunks):
        start = i * step
        result[i] = signal[start:start + chunksize]
        
    return result

def align_references(bases_per_point,dna_sequence, signal_length, chunksize, overlap): 
    
    dna_encoded = encode_dna_sequence(dna_sequence) 
    step = chunksize - overlap

    raw_references = []
    reference_lengths = []
    
    for i in range(0, signal_length - chunksize + 1, step):
        start = int(i * bases_per_point)
        end = int((i + chunksize) * bases_per_point)

        chunk_ref = dna_encoded[start:end]
        
        reference_lengths.append(len(chunk_ref))
        raw_references.append(chunk_ref[::-1])

    max_len = max(reference_lengths)
    padded_references = [np.pad(ref, (0, max_len - len(ref)), 'constant', constant_values=0) for ref in raw_references] 

    return np.array(padded_references, dtype=np.int8), np.array(reference_lengths, dtype=np.int64)

def process_batch_in_parallel(batch_seqs, lut, lambda_time, sampling_rate, output_dir, batch_idx):
    all_chunks, all_refs, all_ref_lens = [None] * len(batch_seqs), [None] * len(batch_seqs), [None] * len(batch_seqs)

    for seq_i, seq in enumerate(batch_seqs):
        signal_df = predict_DNA_6mer_5_3_with_sampling(seq, lut, lambda_time, sampling_rate)
        signal = signal_df['current'].values
        sig_len = len(signal)
        dna_len = len(seq)
        chunksize, overlap = 10000, 500 # chunk size and overlap values changed here
        bases_per_point = 0.05
        
        step = chunksize - overlap
        n_chunks = (sig_len - overlap) // step
        retained_signal_length = n_chunks * step + overlap
        trimmed_signal = signal[:retained_signal_length]
        normalized_signal = normalize_signal(trimmed_signal)
        chunks = chunk_signal_optimized(n_chunks, normalized_signal, chunksize, overlap)
        trimmed_seq_length = dna_len-int(retained_signal_length * bases_per_point)
        trimmed_seq = seq[trimmed_seq_length:]
        refs, ref_lens = align_references(bases_per_point, trimmed_seq, len(trimmed_signal), chunksize, overlap)

        all_chunks[seq_i] = chunks
        all_ref_lens[seq_i] = ref_lens
        all_refs[seq_i] = refs

    all_refs = [ref[::-1] for ref in all_refs]  
    all_ref_lens = [lens[::-1] for lens in all_ref_lens]
    
    max_ref_len = int(max(np.concatenate(all_ref_lens)))
    padded = []
    for ref in all_refs:
        padded_ref = np.array([
            np.pad(chunk, (0, max_ref_len - len(chunk)), 'constant', constant_values=0) 
            for chunk in ref
        ], dtype=np.uint8)
        padded.append(padded_ref)
    
    output_file = os.path.join(output_dir, f"batch_{batch_idx}.npz")
    np.savez_compressed(output_file,
                        chunks=np.concatenate(all_chunks),
                        references=np.concatenate(padded),
                        reference_lengths=np.concatenate(all_ref_lens),
                        max_ref_len=max_ref_len)
    del all_chunks, all_refs, all_ref_lens
    gc.collect()
    return output_file

def process_and_save_batches(sequences, lut, output_dir, prefix, lambda_time, sampling_rate, batch_size):
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    global_max_ref_len = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor: # number of workers can be changed here 
        futures = {}
        for batch_idx, start in enumerate(range(0, len(sequences), batch_size)):
            end = min(start + batch_size, len(sequences))
            batch_seqs = sequences[start:end]
            future = executor.submit(
                process_batch_in_parallel,
                batch_seqs, lut, lambda_time, sampling_rate, output_dir, batch_idx
            )
            futures[future] = batch_idx

        for future in concurrent.futures.as_completed(futures):
            batch_idx = futures[future]
            try:
                output_file = future.result()
                saved_files.append(output_file)

                with np.load(output_file) as data:
                    global_max_ref_len = max(global_max_ref_len, int(data["max_ref_len"]))
                print_memory_usage(f"After batch {batch_idx}")
            except Exception as e:
                print(f"Failed batch {batch_idx}: {e}")
            gc.collect()
