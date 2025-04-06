import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from datasets import load_dataset

def parse_arguments():
    """
    Parse command line arguments

    Returns:
         args (Namespace): parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_name', type=str,
                        help='Name of the dataset from ')

    parser.add_argument('new_dataset_name', type=str,
                        help='How will be called new dataset')

    parser.add_argument('formal_text_column_name', type=str,
                        help='Column with formal texts')

    parser.add_argument('informal_text_column_name', type=str,
                        help='Column with informal texts')

    return parser.parse_args()


def main():
    """
    Main function for cleaning existing dataset
    """
    args = parse_arguments()

    dataset_path = Path(__file__).parent / "datasets"

    # external_dataset_path = dataset_path / "external_datasets" / args.dataset_name
    df = load_dataset(args.dataset_name)['train'].to_pandas()

    formal_data = []
    informal_data = []

    if args.formal_text_column_name == "None" and args.informal_text_column_name == "None":
        print("At least one column should be selected!")
        return 1

    if args.formal_text_column_name != "None":
        formal_data = df[args.formal_text_column_name].values

    formal_labels = np.array([1] * len(formal_data))

    if args.informal_text_column_name != "None":
        informal_data = df[args.informal_text_column_name].values

    informal_labels = np.array([0] * len(informal_data))

    res_texts = np.concatenate((formal_data, informal_data))
    res_labels = np.concatenate((formal_labels, informal_labels))

    new_df = pd.DataFrame({
        'text': res_texts,
        'label': res_labels
    })

    new_df = new_df.sample(frac=1).reset_index(drop=True)

    new_dataset_path = dataset_path / "generated_datasets" / args.new_dataset_name

    new_df.to_csv(new_dataset_path, index=False)

    print(f"Dataset created and saved by path: src/generated_dataset/{args.new_dataset_name}")

if __name__ == '__main__':
    main()