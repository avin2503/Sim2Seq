import os
import pandas as pd
import numpy as np
import uuid
import random
import datetime
import pod5
from pod5.pod5_types import ShiftScalePair

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
    # if you want to convert relative current to pA, uncomment and adjust:
    # signal = signal * 180
    return signal

def convert_to_pod5(input_folder, processed_folder, pod5_output_path, mapping_file):
    os.makedirs(processed_folder, exist_ok=True)

    valid_files = []
    for filename in os.listdir(input_folder):
        if filename.startswith("JS") and filename.endswith(".csv"):
            input_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_csv(input_path)
            except pd.errors.EmptyDataError:
                print(f"Skipped empty file: {filename}")
                continue

            if df.empty or "RelativeCurrent" not in df.columns:
                print(f"Skipped: {filename} (empty or missing 'RelativeCurrent')")
                continue

            df["RelativeCurrent"] = process_signal(df["RelativeCurrent"])

            processed_path = os.path.join(processed_folder, filename)
            df.to_csv(processed_path, index=False)
            valid_files.append(processed_path)
            print(f"Processed and saved: {filename}")

    if not valid_files:
        raise ValueError("No valid CSV files to convert.")

    sample_rate = 5000
    read_id_map = []

    with pod5.Writer(pod5_output_path) as writer:
        for idx, csv_path in enumerate(valid_files, start=1):
            df = pd.read_csv(csv_path)

            if 'RelativeCurrent' not in df.columns:
                print(f"Skipping {csv_path}: missing 'RelativeCurrent'")
                continue

            signal = df['RelativeCurrent'].dropna().values.astype(np.float32)
            if len(signal) == 0:
                print(f"Skipping {csv_path}: empty signal")
                continue

            read_id = uuid.uuid4()
            read_id_map.append((os.path.basename(csv_path), str(read_id)))

            pore = pod5.Pore(
                channel=random.randint(1, 512),
                well=random.randint(1, 4),
                pore_type="R10.4"
            )

            run_info = pod5.RunInfo(
                acquisition_id=str(uuid.uuid4()),
                acquisition_start_time=datetime.datetime.now(),
                adc_min=-32768,
                adc_max=32767,
                context_tags=generate_random_context_tags(sample_rate_hz=sample_rate),
                experiment_name="experiment_name",
                flow_cell_id=str(uuid.uuid4()),
                flow_cell_product_code="FCP001",
                protocol_name="protocol_name",
                protocol_run_id=str(uuid.uuid4()),
                protocol_start_time=datetime.datetime.now(),
                sample_id="sample_id",
                sample_rate=sample_rate,
                sequencing_kit="SQK-LSK110",
                sequencer_position="1",
                sequencer_position_type="type_A",
                software="Software_X",
                system_name="MinION",
                system_type="ONT",
                tracking_id=generate_random_tracking_id()
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
                calibration=pod5.Calibration(offset=0, scale=1),
                pore=pore,
                run_info=run_info,
                end_reason=random_end_reason(),
                predicted_scaling=ShiftScalePair(shift=0, scale=1),
                tracked_scaling=ShiftScalePair(shift=0, scale=1)
            )

            writer.add_read(read)
            print(f"Added read from {os.path.basename(csv_path)} to POD5.")

    mapping_df = pd.DataFrame(read_id_map, columns=["original_file", "read_id"])
    mapping_df.to_csv(mapping_file, index=False)
    print(f"\nRead ID mapping saved to {mapping_file}")
    print(f"\nPOD5 conversion complete: {pod5_output_path}")

def main():
    import sys
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <input_folder> <processed_folder> <pod5_output_path> <mapping_file>")
        sys.exit(1)

    input_folder = sys.argv[1]
    processed_folder = sys.argv[2]
    pod5_output_path = sys.argv[3]
    mapping_file = sys.argv[4]

    convert_to_pod5(input_folder, processed_folder, pod5_output_path, mapping_file)

if __name__ == "__main__":
    main()

