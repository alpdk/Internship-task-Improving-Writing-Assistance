import argparse

import pandas as pd

from transformers import pipeline
from sklearn.utils import shuffle
from huggingface_hub import login

def parse_arguments():
    """
    Parse command line arguments

    Returns:
         args (Namespace): parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('huggingface_token', type=str,
                        help='Token from huggingface hub')

    parser.add_argument('model_name', type=str,
                        help='Name of the model from huggingface')

    parser.add_argument('formal_text_amount', type=int,
                        help='Amount of formal texts in dataset')

    parser.add_argument('informal_text_amount', type=int,
                        help='Amount of informal texts in dataset')

    parser.add_argument('dataset_name', type=str,
                        help='Name of the dataset for saving it')

    return parser.parse_args()

def generate_text(model, prompt, num_samples, label):
    """
    Method to generate texts by prompt

    Parameters:
        model (Pipeline): model to generate texts
        prompt (str): prompt to generate texts
        num_samples (int): number of texts to generate
        label (int): formal or informal label

    Returns:
        res (list): list of generated texts
    """
    res = []

    for _ in range(num_samples):
        model_chain = model(prompt, max_length=100, truncation=True)
        res.append({"text": model_chain[0]['generated_text'], "label": label})

    return res


def main():
    """
    Main function for creating the dataset
    """
    args = parse_arguments()
    login(token=args.huggingface_token)
    model = pipeline('text-generation', model=args.model_name, device=-1, trust_remote_code=True)

    formal_texts = generate_text(model, "Write a formal English sentence using appropriate language.", args.formal_text_amount, 1)
    informal_texts = generate_text(model, "Write an informal English sentence using appropriate language.", args.informal_text_amount, 0)

    data = formal_texts + informal_texts

    df = pd.DataFrame(data)
    df = shuffle(df)

    df.to_csv(f"src/datasets/{args.dataset_name}.csv", index=False)

    print(f"Dataset saved to src/datasets/{args.dataset_name}.csv")

if __name__ == '__main__':
    main()