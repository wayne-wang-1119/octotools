import json
import prettytable
import random
from collections import defaultdict
import argparse
import os 
import re

MODEL_ANSWER_KEY = 'Prediction'


def normalize_answer(answer):
    # Mapping for multiple representations to a standard form
    answer_mapping = {
        'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f',
        '1': 'a', '2': 'b', '3': 'c', '4': 'd', '5': 'e', '6': 'f',
        'a.': 'a', 'b.': 'b', 'c.': 'c', 'd.': 'd', 'e.': 'e', 'f.': 'f',
        '(a)': 'a', '(b)': 'b', '(c)': 'c', '(d)': 'd', '(e)': 'e', '(f)': 'f',
        '"a"': 'a', '"b"': 'b', '"c"': 'c', '"d"': 'd', '"e"': 'e', '"f"': 'f',
        "'a'": 'a', "'b'": 'b', "'c'": 'c', "'d'": 'd', "'e'": 'e', "'f'": 'f',
        'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd', 'E': 'e', 'F': 'f',
        'A.': 'a', 'B.': 'b', 'C.': 'c', 'D.': 'd', 'E.': 'e', 'F.': 'f'
    }

    # Clean and normalize the answer

    normalized = answer.strip().lower()  # Trim and convert to lowercase
    normalized = normalized.strip('.()\'"')  # Remove surrounding punctuation or quotes
    return answer_mapping.get(normalized, normalized)  # Map to standard form or return as is

def eval_string_based(response_text):
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([a-f])"
    
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
    extracted_answer = match.group(1) if match else None
    return extracted_answer

def is_correct(prediction, ground_truth):

    
    prediction_updated = eval_string_based(prediction)

    if prediction_updated == None:
        prediction_updated = prediction

    normalized_pred = normalize_answer(prediction_updated)
    normalized_gt = normalize_answer(ground_truth)

    # Check for exact match or if the prediction starts with the normalized answer
    return normalized_pred == normalized_gt or normalized_pred.startswith(normalized_gt)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract and score the results from the benchmark data")
    parser.add_argument("--result_dir", type=str, default=None, help="The directory containing the results")
    parser.add_argument("--response_type", 
                        type=str, 
                        default="direct_output", 
                        choices=["final_output", "direct_output", "base_response"],
                        help="The type of response to extract from the results")
    
    args = parser.parse_args()

    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    with open('./data/test.json', 'r') as f:
        benchmark_data = json.load(f)
        # convert the benchmark data to a dictionary
        benchmark_data = {data['pid']: data for data in benchmark_data}


    # Load the results
    datas = []
    for file in os.listdir(args.result_dir):
        if file.endswith(".json") and "output_" in file:
            with open(os.path.join(args.result_dir, file), 'r') as f:
                result = json.load(f)
            pid = result["pid"]
            result = {"model_prediction": result[args.response_type]} 
            result.update(benchmark_data[pid])
            datas.append(result)


    for item in datas:
        model_answer = item['model_prediction']

        answer = item["answer"]
        category = item["category"]



        if is_correct(model_answer, answer):
            category_correct[category] += 1

        category_total[category] += 1

    total_correct = 0
    total = 0

    for category in category_total:
        total_correct += category_correct[category]
        total += category_total[category]

    print("Accuracy: ", total_correct / total)
    print("Total: ", total)

    table = prettytable.PrettyTable()
    table.field_names = ["Category", "Total", MODEL_ANSWER_KEY]
    for category in category_total:
        table.add_row([category, category_total[category],
                         category_correct[category] / category_total[category]
                         ])
    #sort by total
    table.sortby = "Total"
    table.reversesort = True
    print(table)


  