import numpy as np
import pandas as pd
import os
import sys
from Bio import SeqIO
import random
import datetime
import pod5
from pod5.pod5_types import ShiftScalePair
import uuid

def moving_6mer_Substrings(string):
    return [string[i:i+6] for i in range(len(string) - 5)]

def predict_DNA_6mer_5_3_with_sampling(template, lut, lambda_time, sampling_rate, I_max=180):
    template = template[::-1]
    kmers = moving_6mer_Substrings(template)
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

        t_pre = np.linspace(current_time, current_time + step_times[i], ns)
        sampled_signals.extend(pre)
        sampled_times.extend(t_pre)
        current_time += step_times[i]

        t_post = np.linspace(current_time, current_time + step_times[i], ns)
        sampled_signals.extend(post)
        sampled_times.extend(t_post)
        current_time += step_times[i]

    return pd.DataFrame({"time": sampled_times, "current": sampled_signals})

def extract_random_sequences(reference_genome_file, num_sequences, min_length=500, max_length=1500):
    genome = "".join(str(record.seq) for record in SeqIO.parse(reference_genome_file, "fasta"))
    genome_len = len(genome)
    if genome_len < max_length:
        raise ValueError("Genome too short for requested max length.")
    sequences = set()
    while len(sequences) < num_sequences:
        length = random.randint(min_length, max_length)
        start = random.randint(0, genome_len - length)
        seq = genome[start:start + length]
        sequences.add(seq)
    return list(sequences)

def generate_multi_read_csv(reference_genome_file, lut_file, num_reads, lambda_time, sampling_rate, output_dir, min_length=200, max_length=500):
    os.makedirs(output_dir, exist_ok=True)
    lut_df = pd.read_csv(lut_file, encoding='utf-8')
    lut = lut_df.set_index("kmer_pull_3_5")[["pre_mean", "pre_std", "post_mean", "post_std"]].to_dict("index")

    sequences = extract_random_sequences(reference_genome_file, num_reads, min_length, max_length)

    mapping = []
    for i, seq in enumerate(sequences):
        signal = predict_DNA_6mer_5_3_with_sampling(seq, lut, lambda_time, sampling_rate)
        out_file = os.path.join(output_dir, f"signal_read_{i+1}.csv")
        signal.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")
        mapping.append((i+1, seq))

    seq_df = pd.DataFrame(mapping, columns=["read_number", "sequence"])
    seq_df.to_csv(os.path.join(output_dir, "sequences.csv"), index=False)
    print(f"Saved sequences.csv in {output_dir}")

def process_signal(signal):
    return signal  # add transformation if needed

def final_conversion(input_folder, processed_folder, pod5_output_path):
    os.makedirs(processed_folder, exist_ok=True)
    valid_files = []

    for filename in os.listdir(input_folder):
        if filename.startswith("signal_read") and filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_folder, filename))
            if df.empty or "current" not in df.columns:
                print(f"Skipped {filename}")
                continue
            df["current"] = process_signal(df["current"])
            out_path = os.path.join(processed_folder, filename)
            df.to_csv(out_path, index=False)
            valid_files.append(out_path)
            print(f"Processed: {out_path}")

    if not valid_files:
        raise ValueError("No valid files to convert.")

    read_id_map = []
    with pod5.Writer(pod5_output_path) as writer:
        for idx, path in enumerate(valid_files, 1):
            df = pd.read_csv(path)
            signal = df["current"].dropna().values.astype(np.float32)
            if len(signal) == 0:
                continue

            read_id = uuid.uuid4()
            read_id_map.append((os.path.basename(path), str(read_id)))

            read = pod5.Read(
                read_id=read_id,
                signal=signal,
                read_number=idx,
                start_sample=0,
                median_before=np.median(signal),
                calibration=pod5.Calibration(0, 1),
                pore=pod5.Pore(random.randint(1,512), random.randint(1,4), "R10.4"),
                run_info=pod5.RunInfo(
                    str(uuid.uuid4()), datetime.datetime.now(), -32768, 32767,
                    {"sample_frequency": "5000"}, "exp", str(uuid.uuid4()),
                    "FCP001", "protocol", str(uuid.uuid4()), datetime.datetime.now(),
                    "sample", 5000, "SQK-LSK110", "1", "type_A",
                    "Software_X", "MinION", "ONT", {"track": "id"}
                ),
                end_reason=pod5.EndReason(pod5.EndReasonEnum.SIGNAL_POSITIVE),
                predicted_scaling=ShiftScalePair(0,1),
                tracked_scaling=ShiftScalePair(0,1)
            )

            writer.add_read(read)
            print(f"Added {os.path.basename(path)} to POD5")

    map_df = pd.DataFrame(read_id_map, columns=["file", "read_id"])
    map_df.to_csv(os.path.join(processed_folder, "real_events_all.csv"), index=False)
    print(f"Saved mapping CSV")

def main():
    if len(sys.argv) != 7:
        print(f"Usage: {sys.argv[0]} <lut_file> <ref_file> <output_dir> <processed_dir> <pod5_output> <num_reads>")
        sys.exit(1)

    lut_file = sys.argv[1]
    ref_file = sys.argv[2]
    output_dir = sys.argv[3]
    processed_dir = sys.argv[4]
    pod5_output = sys.argv[5]
    num_reads = int(sys.argv[6])

    generate_multi_read_csv(ref_file, lut_file, num_reads, 0.002, 5000, output_dir)
    final_conversion(output_dir, processed_dir, pod5_output)

if __name__ == "__main__":
    main()
