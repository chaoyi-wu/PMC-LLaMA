'''
python eval_medqa.py \
    --result-dir /path/to/inferenced_result_dir \
    --write-path /path/to/extracted_answers.json
'''

import os
import argparse
from typing import Sequence
from tqdm import tqdm
import re
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str)
    parser.add_argument('--write-path', type=str)
    args = parser.parse_args()
    return args


def traversal_files(dir_path: str, extension: str):
    file_names = []

    for directory, dirnames, filenames in os.walk(dir_path):
        for file_name in filenames:
            if file_name.endswith(extension):
                file_names.append(file_name)

    return file_names


def parse_pmc_answers(result_dir: str, file_names: Sequence[str]):
    pmc_answers = {}
    no_answer_num = 0

    for file_name in tqdm(file_names):
        sample_id = file_name.split('.')[0]
        sample_id = int(sample_id)

        file_path = os.path.join(result_dir, file_name)
        with open(file_path, 'r') as f:
            answer_str = f.read()
        matched_pieces = re.findall(r'(?i)OPTION [ABCD] IS CORRECT', answer_str)  # could have more than one pieces of matched str

        if len( matched_pieces ) == 0:  # no matched piece
            no_answer_num += 1
            continue
        predicted_option = matched_pieces[0].split()[1]

        pmc_answers[sample_id] = predicted_option
    # endfor

    # raise RuntimeError( len(pmc_answers) )
    print(f"\033[32mNo Answer Num\033[0m: {no_answer_num}")
    return pmc_answers


if __name__ == '__main__':

    args = parse_args()
    answer_file_names = traversal_files(args.result_dir, 'txt')
    predicted_answers = parse_pmc_answers(args.result_dir, answer_file_names)

    with open(args.write_path, 'w') as f:
        json.dump(predicted_answers, f)

