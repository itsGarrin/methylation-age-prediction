import os

import pandas as pd


def read_series_matrix_in_chunks(file_path, chunk_size=100):
    """Read a Series Matrix file in chunks and return the combined DataFrame."""
    chunk_list = []  # Store chunks in a list
    for chunk in pd.read_csv(file_path, sep='\t', header=0, index_col=0, comment='!', chunksize=chunk_size):
        chunk_list.append(chunk)
    # Combine all chunks into a single DataFrame
    combined_data = pd.concat(chunk_list, axis=1)
    return combined_data


def combine_datasets(folder_path, chunk_size=1000):
    """Combine all Series Matrix files in a folder into a single DataFrame by reading in chunks."""
    combined_data = None
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.txt'):
            print(f"Reading: {file}")
            file_path = os.path.join(folder_path, file)
            # Read the file in chunks and combine
            data = read_series_matrix_in_chunks(file_path, chunk_size)
            combined_data = data if combined_data is None else pd.concat([combined_data, data], axis=1)
    return combined_data


def preprocess_and_combine(folder_27k, folder_450k, output_path, chunk_size=1000):
    # Combine all datasets within each folder
    print("Combining 27k datasets...")
    combined_27k = combine_datasets(folder_27k, chunk_size)
    print(f"Combined 27k shape: {combined_27k.shape}")

    print("Combining 450k datasets...")
    combined_450k = combine_datasets(folder_450k, chunk_size)
    print(f"Combined 450k shape: {combined_450k.shape}")

    # Find common CpG probes
    common_probes = combined_27k.index.intersection(combined_450k.index)
    print(f"Number of overlapping probes: {len(common_probes)}")

    # Filter to overlapping probes
    combined_27k = combined_27k.loc[common_probes]
    combined_450k = combined_450k.loc[common_probes]

    # Concatenate datasets (merge 27k and 450k)
    combined_data = pd.concat([combined_27k, combined_450k], axis=1)

    # Ensure sample consistency (check if both datasets have the same samples)
    if not combined_27k.columns.equals(combined_450k.columns):
        print("Warning: Sample order between 27k and 450k does not match!")
        # Optional: Implement sample alignment here if needed

    # Filter samples with high inter-array correlation
    sample_corr = combined_data.corr().mean(axis=1)
    high_quality_samples = sample_corr[sample_corr > 0.90].index
    combined_data = combined_data[high_quality_samples]
    print(f"Number of high-quality samples: {len(high_quality_samples)}")

    # Impute missing values with column means
    combined_data.fillna(combined_data.mean(), inplace=True)

    # Save the final combined dataset
    combined_data.to_csv(output_path)
    print(f"Final preprocessed dataset saved to {output_path}")


# Main function to encapsulate the script execution
def main():
    # Paths to folders and output
    folder_27k = "../data/27k"
    folder_450k = "../data/450k"
    output_path = "../data/methylation.csv"

    # Preprocess and combine datasets
    preprocess_and_combine(folder_27k, folder_450k, output_path)


# Ensure this script only runs when executed directly (not when imported)
if __name__ == "__main__":
    main()
