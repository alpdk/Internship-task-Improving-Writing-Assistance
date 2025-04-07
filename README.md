# Identification of the text formality

## How text will be classified

This repository contains code for classification of text on formal and informal.
The main parts of the code are:

1. Formality approach: this approaches represent 3 ways to classify text on formal and informal.
   1. Model free approach: there used check on behaviours, that represented in formal and informal texts.
            For example: formal text do not contains constrains and slang words.

   2. HuggingFace model approach: using models from huggingface locally. 
      This models should be specifically train for text-classification task and should have _'label'_ field with answers: _formal_ and _informal_.
   3. Gemini approach: there we making prompt into gemini 2.0 flash apo with request to classify our text as formal or informal text.
2. Metrics: different metrics for identification how good is our approach. There were implemented:
   1. Accuracy
   2. Recall
   3. Precision
   4. F1Score
   5. SumMetric: this metric calculate sum of the other metrics multiplied by coefficients ([0.1, 0.2, 0.4, 0.3]). 
      This metric help us to understand which model best in compression with other and mainly concentrated on do not select approach with false positive outcomes.

This classes will help us to make different experiments with approach combinations and metrics to select the best approach.

## Dataset

Current dataset contains 2 parts: [dataset from huggingface](https://huggingface.co/datasets/Mehaki/Formal_to_Casual-1), and generated dataset with usage of [mistralai
/
Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

Huggingface dataset contains almost 100 formal and 100 informal text, while generated contains 20 example of each text in languages: 
__German__, __English__, __French__, __Spanish__, __Russian__.

If you want to generate your own datasets you should run this lines with your replacements:

```
python prepare_existing_dataset "DATASET_NAME_FROM_HUGGINGFACE" "OUTPUT_FILE_NAME" "FORMAL_TEXT_COLUMN" "INFORMAL_TEXT_COLUMN"
```

```
python generate_text_dataset "HUGGING_FACE_TOKEN" "MODEL_NAME_FROM_HUGGINGFACE" "FINAL_COUNT_OF_FORMAL_TEXTS" "FINAL_COUNT_OF_INFORMAL_TEXTS" "OUTPUT_FILE_NAME"
```

If you want to combine datasets, you should give them different names and after that merge them.

I wanted to use different methods, because I did not found good labeled datasets for this task, and we need not only 1 language for test.

## Evaluation

Main script of this project is _test_approach.py_. This script evaluate all 5 metrics described in previous part for specific formality approach.

There is code, how to use it:

```
python test_approach.py "DATASET_NAME" "RESULT_TABLE_NAME" "APPROACH_NAME" "--MODEL_NAME" "--API_KEY_OR_HF_TOKEN"
```

Last 2 parameters are required only for model based approach, and gemini approach.

There I wanted to compare results between different approaches, because all of them have pros and cons, that will be described in next chapter. 

Also, if you want to run it in colab, you can run it at this [notebook](https://colab.research.google.com/drive/1QenYk33Ws-PEhND6Shfv-SzsvtH3V8t7?usp=sharing).
But you will need to select one of the templates for _outside_args_ in first cell of the _Test_ part with suitable arguments for you.

## Results

I was able to collect data only with first 2 approaches, because I could not take full dataset evaluation from gemini.
But I can say, that this approach have a huge pros in comfort, because it do not require to make strong prompt engineering as it was with smaller model from huggingface.
Also, we can say that using model through api helps to free memory, while first to approaches was run locally.
As a cons, we can remember that we do not have enough control over model, and it can be costly for huge datasets.

| Approach                                          | Accuracy | F1Score | Precision | Recall | SumMetric |
|---------------------------------------------------|----------|---------|-----------|--------|-----------|
| ModelFreeApproach                                 | 0.54     | 0.13    | 1.0       | 0.07   | 0.5       |
| LenDigLearn/formality-classifier-mdeberta-v3-base | 0.5      | 0.67    | 0.5       | 1.0    | 0.65      |

There we can see that ModelFreeApproach has very bad prediction values. 
First of all, this approach was implemented only for English language, and it will not be working with other languages as it should.
Secondly, this approach require a lot of time for analyzing language and his rules for separating text on formal and informal.
Finally, implementation of this approach require strong understanding of which parameter is more important the other for separation.
We can say that this approach give us full control over the system, and it can take less resources in work, but it hard to create proper system for different languages.

Approach with usage of the specially fine-tuned model show us much better results in compression with the model free approach.
Firstly, it was trained on different languages, that do not require different models for different languages.
Secondly, it is easier to train for different languages, because we do not need to understand every aspect of the language.
We can say, that this approach can take much more resources, because of the model size, but as a result we have a tool for different languages.

## Challenges

One of the biggest challenges is to make proper prompt for the model. I was experimenting with text-to-text model from huggingface, but prompt should be very good for specific model.

Also, I could not find good and big datasets in opensource, and usually their are closed.