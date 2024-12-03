import os

import pandas as pd


def read_series_matrix_in_chunks(file_path, chunk_size=1000, output_path=None, first_chunk=False):
    """Read a Series Matrix file (.txt) in chunks and write it directly to disk."""
    for chunk in pd.read_csv(file_path, sep='\t', header=0, index_col=0, comment='!', chunksize=chunk_size):
        if first_chunk:
            # Write the first chunk to the output file (overwrite/create file)
            chunk.to_csv(output_path, mode='w', header=True)
            first_chunk = False
        else:
            # Append subsequent chunks to the same file
            chunk.to_csv(output_path, mode='a', header=False)


def combine_datasets(folder_path, output_path, chunk_size=1000):
    """Combine all Series Matrix files in a folder into a single output file by reading in chunks."""
    first_chunk = True  # Keeps track of whether we are writing the first chunk or appending

    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.txt'):  # Process only .txt files
            print(f"Reading: {file}")
            file_path = os.path.join(folder_path, file)

            # Process and write chunks to the output file
            read_series_matrix_in_chunks(file_path, chunk_size, output_path, first_chunk)
            first_chunk = False  # After first file, switch to append mode


def preprocess_and_combine(folder_27k, folder_450k, output_27k, output_450k, final_output, chunk_size=1000):
    """Preprocess and combine the 27k and 450k datasets into separate outputs, then merge on common probes."""
    print("Combining 27k datasets...")
    #combine_datasets(folder_27k, output_27k, chunk_size)
    print(f"Combined 27k saved to: {output_27k}")

    print("Combining 450k datasets...")
    #combine_datasets(folder_450k, output_450k, chunk_size)
    print(f"Combined 450k saved to: {output_450k}")

    # Load the combined 27k and 450k data
    print("Loading combined datasets...")
    combined_27k = pd.read_csv(output_27k, index_col=0, on_bad_lines='skip')
    combined_450k = pd.read_csv(output_450k, index_col=0, on_bad_lines='skip')

    # Aggregate duplicate indices by taking the mean
    combined_27k = combined_27k.groupby(combined_27k.index).mean()
    combined_450k = combined_450k.groupby(combined_450k.index).mean()

    # Find common CpG probes
    print("Finding common probes...")
    common_probes = combined_27k.index.intersection(combined_450k.index)
    print(f"Number of overlapping probes: {len(common_probes)}")

    # Filter both datasets to keep only the common probes
    combined_27k = combined_27k.loc[common_probes]
    combined_450k = combined_450k.loc[common_probes]

    # Concatenate the two datasets
    print("Merging datasets...")
    combined_data = pd.concat([combined_27k, combined_450k], axis=1)

    # Filter samples with high inter-array correlation
    '''
    print("Filtering high-quality samples...")
    sample_corr = combined_data.corr().mean(axis=1)
    high_quality_samples = sample_corr[sample_corr > 0.90].index
    combined_data = combined_data[high_quality_samples]
    print(f"Number of high-quality samples: {len(high_quality_samples)}")
    '''

    # Impute missing values with column means
    print("Imputing missing values...")
    combined_data.fillna(combined_data.mean(), inplace=True)

    # Save the final processed dataset
    combined_data.to_csv(final_output)
    print(f"Final preprocessed dataset saved to: {final_output}")


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
