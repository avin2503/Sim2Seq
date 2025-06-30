# Sim2Seq: developing DNA basecalling for MspA+Hel308 nanopore sequencing 

Author:  
**Avin Nayeri**

Project:  
**Bachelor End Project**

Degree:  
**Bachelor of Science in Nanobiology**

Institutions:  
**TU Delft**  
**Erasmus MC**

This code was developed as part of a Bachelor End Project thesis for the Nanobiology program at TU Delft and Erasmus MC, aiming to design and implement a DNA basecalling pipeline for MspA+Hel308 nanopore sequencing.

## Training Data Generation

Sim2Seq provides three nanopore signal generators: SimGen_Ideal.py, SimGen_Noise.py, and SimGen_Realistic.py. In my thesis, the signal generation and training data preparation were described as separate components. However, for simplicity and usability, these have been integrated. Each of these files performs two key tasks: (1) generates synthetic nanopore current signals based on a provided reference DNA genome and a 6-mer signal prediction model and (2) automatically converts the simulated signals into the required .npy files (chunks, references, and reference_lengths) for training a Bonito basecalling model from scratch.

To note, SG_reference_genome.fna (the reference genome of Streptococcus dysgalactiae) and DNA_6mer_prediction_model.csv (the k-mer model provided by the CD Lab) were used for this thesis. These can be swapped for other files if needed, but must ensure that the file formats and structures are compatible. 

### Usage template for the signal generators

```python

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
