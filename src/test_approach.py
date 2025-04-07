import argparse
import pandas as pd

from pathlib import Path

from src.formality_approaches.ModelFreeApproach import ModelFreeApproach
from src.formality_approaches.HuggingFaceModelApproach import HuggingFaceModelApproach
from src.formality_approaches.GeminiApproach import GeminiApproach

from metrics.Accuracy import Accuracy
from metrics.Recall import Recall
from metrics.F1Score import F1Score
from metrics.Precision import Precision
from metrics.SumMetric import SumMetric


def parse_arguments():
    """
    Parse command line arguments

    Returns:
         args (Namespace): parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_name', type=str,
                        help='Name of the dataset from ')

    parser.add_argument('output_file_name', type=str,
                        help='Where will be saved results')

    parser.add_argument('approach_name', type=str,
                        help='What approach will be used for evaluation')

    parser.add_argument('--model_name', type=str,
                        help='Name of the model from huggingface')

    parser.add_argument('--token', type=str,
                        help='token for the model')

    return parser.parse_args()

def load_approach(args):
    """
    Method for loading the formality approach

    Parameters:
        args (Namespace): parsed command line arguments

    Returns:
        res (ApproachTemplate): approach that will be used
    """

    approach_name = args.approach_name.lower()

    match approach_name:
        case "modelfreeapproach":
            return ModelFreeApproach()
        case "huggingfacemodelapproach":
            return HuggingFaceModelApproach(args.model_name, args.token)
        case "geminiapproach":
            return GeminiApproach()
        case _:
            return None

def main():
    args = parse_arguments()

    datasets_path = Path(__file__).parent / "datasets"

    datasets_path = datasets_path / "generated_datasets" / args.dataset_name

    df = pd.read_csv(datasets_path)

    approach = load_approach(args)

    metrics = [
        Accuracy(args.dataset_name, df, approach),
        F1Score(args.dataset_name, df, approach),
        Precision(args.dataset_name, df, approach),
        Recall(args.dataset_name, df, approach),
        SumMetric(args.dataset_name, df, approach)
    ]

    for metric in metrics:
        metric.evaluate()

if __name__ == '__main__':
    main()