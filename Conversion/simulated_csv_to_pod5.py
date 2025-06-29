import numpy as np
import pandas as pd
import os
from Bio import SeqIO
import random
import glob 
import datetime
import pod5 
from pod5.pod5_types import ShiftScalePair, Pore, Calibration, EndReason, RunInfo
import uuid 

def moving_6mer_Substrings(string):
    return [string[i:i+6] for i in range(len(string) - 5)]

def predict_DNA_6mer_5_3_with_sampling(template, lut, lambda_time, sampling_rate, I_max=180): # change this depending on which generator used 
    template = template[::-1] 
    kmers = moving_6mer_Substrings(template)
    N = len(kmers)
    valid_kmers = [k for k in kmers if k in lut]
    params = np.array([list(lut[k].values()) for k in valid_kmers]) 
    pre_mean, pre_std, post_mean, post_std = params.T
    step_times = np.ones(len(valid_kmers)) * lambda_time 
    num_samples = (step_times * sampling_rate).astype(int)

    sampled_signals = []
    sampled_times = []
    current_time = 0.0
    
    for i in range(len(valid_kmers)):
        ns = num_samples[i]
        if ns == 0:
            continue
        
        pre = np.random.normal(pre_mean[i] * I_max, pre_std[i] * I_max, ns)
        post = np.random.normal(post_mean[i] * I_max, post_std[i] * I_max, ns)
        
        step_time = step_times[i]
        
        t_pre = np.linspace(current_time, current_time + step_time, ns)
        sampled_signals.extend(pre)
        sampled_times.extend(t_pre)
        current_time += step_time
       
        t_post = np.linspace(current_time, current_time + step_time, ns)
        sampled_signals.extend(post)
        sampled_times.extend(t_post)
        current_time += step_time

    return pd.DataFrame({
        "time": sampled_times,
        "current": sampled_signals
    })

def extract_random_sequences(reference_genome_file, num_sequences, min_length=500, max_length=1500):
    genome = "".join(str(record.seq) for record in SeqIO.parse(reference_genome_file, "fasta"))
    genome_len = len(genome)
    sequences = set()
    while len(sequences) < num_sequences:
        length = random.randint(min_length, max_length)
        start = random.randint(0, genome_len - length)
        seq = genome[start:start + length]
        sequences.add(seq)
    return list(sequences)


def generate_signal_data(sequences, lut, lambda_time, sampling_rate,output_dir):
    signal_dfs = []
    
    for i, seq in enumerate(sequences):
        signal = predict_DNA_6mer_5_3_with_sampling(seq, lut, lambda_time, sampling_rate)
        signal_df = pd.DataFrame({
            'time': signal['time'],
            'current': signal['current']
        })
        output_file = f"signal_read_{i+1}.csv"
        file_path = os.path.join(output_dir, output_file)
        signal_df.to_csv(file_path, index=False)
        print(f"Saved signal to {output_file}")
        signal_dfs.append((i+1, seq))

    return signal_dfs  

def generate_multi_read_csv(reference_genome_file, lut, num_reads, lambda_time, sampling_rate, output_dir, min_length=500, max_length=1500):
    os.makedirs(output_dir, exist_ok=True)

    total_sequences = extract_random_sequences(reference_genome_file, num_reads, min_length, max_length)

    read_seq_mapping = generate_signal_data(total_sequences, lut, lambda_time, sampling_rate,output_dir)

    sequence_df = pd.DataFrame(read_seq_mapping, columns=["read_number", "sequence"])
    sequence_file = os.path.join(output_dir, "sequences.csv")
    sequence_df.to_csv(sequence_file, index=False)
    print(f"Saved sequence mappings to {sequence_file}")

def random_string(length=8):
    return ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=length))

def generate_random_context_tags(sample_rate_hz=5000):
    return {
        "sample_frequency": str(sample_rate_hz),
        "sequencing_kit": "SQK-LSK110",
        "flow_cell_product_code": "FCP001",
        "protocol_name": "protocol_name",
        "protocol_run_id": str(uuid.uuid4()),
        "sample_id": "sample_id"
    }

def generate_random_tracking_id():
    return {random_string(5): random_string(10) for _ in range(random.randint(1, 3))}

def random_end_reason():
    end_reasons = [
        pod5.EndReason(pod5.EndReasonEnum.SIGNAL_POSITIVE, forced=False),
        pod5.EndReason(pod5.EndReasonEnum.SIGNAL_NEGATIVE, forced=False),
        pod5.EndReason(pod5.EndReasonEnum.SIGNAL_POSITIVE, forced=True),
        pod5.EndReason(pod5.EndReasonEnum.SIGNAL_NEGATIVE, forced=True)
    ]
    return random.choice(end_reasons)

def process_signal(signal):
    return signal # make changes to signal if needed

def generate_final_csv():
    LUT_6mer = pd.read_csv('...', encoding='utf-8') # add kmer model here 
    lut = LUT_6mer.set_index("kmer_pull_3_5")[["pre_mean", "pre_std", "post_mean", "post_std"]].to_dict("index") # change parameter names if needed

    generate_multi_read_csv(
        reference_genome_file='...', # add reference genome file here, preferably .fna format 
        lut=lut,
        num_reads=1000, # change if needed 
        lambda_time=0.002, # change if needed 
        sampling_rate=5000, # change if needed 
        output_dir='...', # add output directory here 
        min_length=200, # change if needed 
        max_length=500 # change if needed 
    )
  
def final_conversion(input_folder, processed_folder, pod5_output_path):
    valid_files = []
    for filename in os.listdir(input_folder):
        if filename.startswith("signal_read") and filename.endswith(".csv"):
            input_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_csv(input_path)
            except pd.errors.EmptyDataError:
                print(f"Skipped empty file: {filename}")
                continue

            if df.empty or "current" not in df.columns:
                print(f"Skipped: {filename} (empty or missing 'current')")
                continue

            df["current"] = process_signal(df["current"])

            processed_path = os.path.join(processed_folder, filename)
            df.to_csv(processed_path, index=False)
            valid_files.append(processed_path)
            print(f"Processed and saved: {filename}")

    sample_rate = 5000 # change if needed

    if not valid_files:
        raise ValueError("No valid CSV files to convert.")

    read_id_map = []
    with pod5.Writer(pod5_output_path) as writer:
        for idx, csv_path in enumerate(valid_files, start=1):
            df = pd.read_csv(csv_path)

            if 'current' not in df.columns:
                print(f"Skipping {csv_path}: missing 'current'")
                continue

            signal = df['current'].dropna().values.astype(np.float32)
            if len(signal) == 0:
                print(f"Skipping {csv_path}: empty signal")
                continue

            read_id = uuid.uuid4()
            read_id_map.append((os.path.basename(csv_path), str(read_id)))
            channel = random.randint(1, 512)
            well = random.randint(1, 4)
            pore = pod5.Pore(channel=channel, well=well, pore_type="R10.4")

            calibration = pod5.Calibration(offset=0, scale=1)
            end_reason = random_end_reason()
            context_tags = generate_random_context_tags(sample_rate_hz=sample_rate)
            tracking_id = generate_random_tracking_id()
            current_time = datetime.datetime.now()

            run_info = pod5.RunInfo(
                acquisition_id=str(uuid.uuid4()),
                acquisition_start_time=current_time,
                adc_min=-32768,
                adc_max=32767,
                context_tags=context_tags,
                experiment_name="experiment_name",
                flow_cell_id=str(uuid.uuid4()),
                flow_cell_product_code="FCP001",
                protocol_name="protocol_name",
                protocol_run_id=str(uuid.uuid4()),
                protocol_start_time=current_time,
                sample_id="sample_id",
                sample_rate=sample_rate,
                sequencing_kit="SQK-LSK110",
                sequencer_position="1",
                sequencer_position_type="type_A",
                software="Software_X",
                system_name="MinION",
                system_type="ONT",
                tracking_id=tracking_id,
            )

            time_col = df['time'].dropna().values if 'time' in df.columns else np.array([])
            start_sample = int(time_col.min() * sample_rate) if len(time_col) > 0 else 0
            median_before = np.median(signal)

            read = pod5.Read(
                read_id=read_id,
                signal=signal,
                read_number=idx,
                start_sample=start_sample,
                median_before=median_before,
                calibration=calibration,
                pore=pore,
                run_info=run_info,
                end_reason=end_reason,
                predicted_scaling=ShiftScalePair(shift=0, scale=1),
                tracked_scaling=ShiftScalePair(shift=0, scale=1)
            )

            writer.add_read(read)
            print(f"Added read from {os.path.basename(csv_path)} to POD5.")

    mapping_df = pd.DataFrame(read_id_map, columns=["original_file", "read_id"])
    sequences_path = os.path.join(input_folder, "sequences.csv")
    sequences_df = pd.read_csv(sequences_path)
    mapping_df["read_number"] = mapping_df["original_file"].str.extract(r"signal_read_(\d+)\.csv").astype(int)
    merged_df = mapping_df.merge(sequences_df, on="read_number", how="left")

    mapping_file = os.path.join(processed_folder, "real_events_all.csv")
    merged_df.to_csv(mapping_file, index=False)
    print(f"\nRead ID mapping with sequences saved to {mapping_file}")
    print(f"\nPOD5 conversion complete: {pod5_output_path}")

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <input_folder> <processed_folder> <pod5_output_path>")
        sys.exit(1)

    input_folder = sys.argv[1]
    processed_folder = sys.argv[2]
    pod5_output_path = sys.argv[3]

    generate_final_csv()
    final_conversion(input_folder, processed_folder, pod5_output_path)

if __name__ == "__main__":
    main()

