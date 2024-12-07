import os
import re

import dask.dataframe as dd
import pandas as pd


def read_and_extract_ages(file_path):
    """Extract ages and GEO accession IDs from the Series Matrix file."""
    ages = {}  # Dictionary to store {GEO Accession ID: Age}
    geo_accessions = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('!Sample_geo_accession'):
                geo_accessions = [id_.strip('"') for id_ in line.strip().split('\t')[1:]]  # Skip the first entry
                print(f"GEO Accessions: {geo_accessions}")
            elif line.startswith('!Sample_characteristics_ch1') and 'age' in line.lower():
                age_matches = re.findall(r'age(?: \(y\))?:\s*(\d+)', line, flags=re.IGNORECASE)
                if age_matches and geo_accessions:
                    ages = dict(zip(geo_accessions, map(int, age_matches)))
                    print(f"Ages: {ages}")
                break
    return ages


def process_and_append(file_path, output_path, first_file):
    """Transpose a Series Matrix file and append it to the output CSV."""
    # Read the full matrix data
    data = pd.read_csv(file_path, sep='\t', header=0, index_col=0, comment='!')

    # Transpose the data so that ID_REF is the index
    data_transposed = data.T
    data_transposed.index.name = 'Sample'

    # Write to CSV, creating the file if it's the first file or appending otherwise
    if first_file:
        data_transposed.to_csv(output_path, mode='w', header=True)
    else:
        data_transposed.to_csv(output_path, mode='a', header=False)  # Append without headers


def combine_datasets(folder_path, output_path):
    """Combine all Series Matrix files by transposing and appending to a single CSV."""
    first_file = True
    all_ages = {}  # Store all sample-to-age mappings

    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.txt'):  # Process only .txt files
            print(f"Processing: {file}")
            file_path = os.path.join(folder_path, file)

            # Extract ages from the file
            ages = read_and_extract_ages(file_path)
            all_ages.update(ages)

            # Process and append data to the CSV
            process_and_append(file_path, output_path, first_file)
            first_file = False  # Subsequent files should append, not overwrite

    # Return a DataFrame with samples and ages
    return pd.DataFrame(list(all_ages.items()), columns=["Sample", "Age"])


def process_with_dask(file_path):
    """Read and process large CSV files in chunks using Dask."""
    # Read the file lazily with Dask (without setting index_col directly)
    ddf = dd.read_csv(file_path,
                      assume_missing=True,
                      sample=10_000_000,  # Increase sample size to 10 MB
                      blocksize="64MB",  # Adjust block size for Dask's partitions
                      on_bad_lines='skip')

    # Set the index manually after loading the CSV
    ddf = ddf.set_index(ddf.columns[0])  # Assuming the first column is the index column (ID_REF)

    # Process the data lazily (e.g., group by probes and aggregate)
    ddf = ddf.groupby(ddf.index).mean()  # Example aggregation for duplicate probes

    return ddf


def preprocess_and_combine(folder_27k, folder_450k, output_27k, output_450k, final_output):
    """Preprocess and combine the 27k and 450k datasets into separate outputs, then merge on common probes."""
    print("Combining 27k datasets and extracting ages...")
    # ages_27k = combine_datasets(folder_27k, output_27k)
    print(f"Combined 27k saved to: {output_27k}")

    print("Combining 450k datasets and extracting ages...")
    # ages_450k = combine_datasets(folder_450k, output_450k)
    print(f"Combined 450k saved to: {output_450k}")

    # Combine extracted ages
    # all_ages = pd.concat([ages_27k, ages_450k], axis=0).drop_duplicates(subset="Sample").reset_index(drop=True)
    all_ages = pd.read_csv("../data/ages.csv")
    print(f"Extracted {len(all_ages)} unique ages.")

    # Save the extracted ages to a file
    ages_output_path = "../data/ages.csv"
    all_ages.to_csv(ages_output_path, index=False)
    print(f"Ages saved to: {ages_output_path}")

    # Use Dask to load combined datasets
    print("Loading combined datasets using Dask...")
    combined_27k_ddf = process_with_dask(output_27k)
    combined_450k_ddf = process_with_dask(output_450k)

    # Debug: Check if datasets were loaded
    print(f"27k dataset size: {len(combined_27k_ddf.columns)} columns, {len(combined_27k_ddf)} rows (Dask LazyFrame)")
    print(
        f"450k dataset size: {len(combined_450k_ddf.columns)} columns, {len(combined_450k_ddf)} rows (Dask LazyFrame)")

    # Find common probes
    print("Finding common probes...")
    common_probes = combined_27k_ddf.index.intersect(combined_450k_ddf.index).compute()
    print(f"Number of overlapping probes: {len(common_probes)}")

    # Filter datasets to common probes
    combined_27k_ddf = combined_27k_ddf.loc[common_probes]
    combined_450k_ddf = combined_450k_ddf.loc[common_probes]

    # Concatenate datasets by probes
    print("Concatenating datasets...")
    combined_data_ddf = dd.concat([combined_27k_ddf, combined_450k_ddf], axis=1)

    # Ensure valid sample names and alignment
    valid_samples = [sample.strip('"').strip() for sample in all_ages["Sample"].tolist()]
    combined_data_ddf = combined_data_ddf[valid_samples]

    # Transpose data
    aligned_data_ddf = combined_data_ddf.T
    aligned_data_ddf["Age"] = all_ages.set_index("Sample").loc[aligned_data_ddf.index, "Age"]

    # Impute missing values with column means
    aligned_data_ddf = aligned_data_ddf.fillna(aligned_data_ddf.mean())

    # Save the final processed dataset
    print("Saving final dataset...")
    aligned_data_ddf.to_csv(final_output, single_file=True)
    print(f"Final preprocessed dataset with aligned ages saved to: {final_output}")


def main():
    # Input directories for the datasets
    folder_27k = "../data/27k"
    folder_450k = "../data/450k"

    # Output files
    output_27k = "../data/combined_27k.csv"
    output_450k = "../data/combined_450k.csv"
    final_output = "../data/final_methylation.csv"

    # Preprocess and combine datasets
    preprocess_and_combine(folder_27k, folder_450k, output_27k, output_450k, final_output)


# Ensure this script only runs when executed directly (not when imported)
if __name__ == "__main__":
    main()
