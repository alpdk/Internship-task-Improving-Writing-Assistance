import argparse
import torch
import numpy as np
import pandas as pd

from transformers import pipeline
from sklearn.utils import shuffle
from huggingface_hub import login
from pathlib import Path


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

    languages = ['German', 'English', 'French', 'Italian', 'Russian']

    for language in languages:
        prompt = (
            f"Generate me please a text with no more than 100 words. Generated text should be written in {language} language with usage of {prompt} style. I want to see ONLY requested text.")

        for _ in range(num_samples // len(languages)):
            model_chain = model(prompt,
                                max_length=200,
                                truncation=True,
                                temperature=0.9,
                                top_p=0.9,
                                do_sample=True)

            response = model_chain[0]['generated_text']

            if response.startswith(prompt):
                response = response[len(prompt):].strip()

            res.append({"text": response, "label": label})

    return res


def main():
    """
    Main function for creating the dataset
    """
    args = parse_arguments()
    login(token=args.huggingface_token)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = pipeline('text-generation', model=args.model_name, device=device, trust_remote_code=True)

    formal_data = generate_text(model, "formal", args.formal_text_amount, 1)
    informal_data = generate_text(model, "informal", args.informal_text_amount, 0)

    formal_labels = np.array([1] * len(formal_data))
    informal_labels = np.array([0] * len(informal_data))

    res_texts = np.concatenate((formal_data, informal_data))
    res_labels = np.concatenate((formal_labels, informal_labels))

    new_df = pd.DataFrame({
        'text': res_texts,
        'label': res_labels
    })

    new_df = shuffle(new_df)

    save_path = Path(__file__).parent / "datasets" / "generated_datasets" / args.dataset_name

    new_df.to_csv(f"{save_path}", index=False)

    print(f"Dataset saved to src/datasets/{args.dataset_name}")


if __name__ == '__main__':
    main()
