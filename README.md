# Sim2Seq: developing DNA basecalling for MspA+Hel308 nanopore sequencing 

## Signal Generator Usage

Sim2Seq includes three signal generators that generate synthetic nanopore current signals from reference DNA sequences using a 6-mer signal prediction model.

### Example code

```python
import pandas as pd

# Reference genome in FASTA (.fna) format
reference_genome = 'path/to/reference_genome.fna'

# Load the 6-mer LUT (lookup table) model
LUT_6mer = pd.read_csv('path/to/DNA_6mer_prediction_model.csv', encoding='utf-8')
lut = LUT_6mer.set_index("kmer_pull_3_5")[["pre_mean", "pre_std", "post_mean", "post_std"]].to_dict("index")

# Generate random training and validation sequences
train_seqs, test_seqs = generate_random_signals(
    reference_genome_file=reference_genome,
    num_reads=1000,         # Total number of reads
    seq_length=400          # Length of each sequence
)

# Output directories
output_dir_train = "output/train"
output_dir_test = "output/train/validation"

# Generate training data signals
process_and_save_batches(
    sequences=train_seqs,
    lut=lut,
    output_dir=output_dir_train,
    prefix="train",
    lambda_time=0.002,      # Dwell time per 6-mer step (in seconds)
    sampling_rate=5000,     # Sampling rate (Hz)
    batch_size=500          # Number of reads per batch
)

# Generate validation data signals
process_and_save_batches(
    sequences=test_seqs,
    lut=lut,
    output_dir=output_dir_test,
    prefix="test",
    lambda_time=0.002,
    sampling_rate=5000,
    batch_size=500
)
