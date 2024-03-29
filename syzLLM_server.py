import random

from flask import Flask, request, jsonify
from transformers import AutoModelForMaskedLM
import torch

from syz_tokenizer import SyzTokenizer
from utils import ModelPath, VocabFilePath, CLS, SEP, UNK_idx, UNK

tokenizer = SyzTokenizer()
mask_model = AutoModelForMaskedLM.from_pretrained(ModelPath)
syscall_name_dict = {}


def extract_syscall_name(syscall):
    if '(' in syscall:
        return syscall[:syscall.index('(')]

    return syscall


def init_env():
    with open(VocabFilePath, 'r') as file:
        for line in file:
            syscall = line.strip()
            syscall_name = extract_syscall_name(syscall)
            syscall_name_dict.setdefault(syscall_name, list()).append(syscall)


def validate_syscall(syscall_list):
    new_syscall_list = []
    for syscall in syscall_list:
        syscall_name = extract_syscall_name(syscall)
        if tokenizer.tokenize_word(syscall) != UNK_idx:
            new_syscall_list.append(syscall)
        elif syscall_name in syscall_name_dict:
            # score?
            syscall_set = syscall_name_dict[syscall_name]
            idx = random.randint(0, len(syscall_set) - 1)
            new_syscall_list.append(syscall_set[idx])
        else:
            new_syscall_list.append(UNK)

    return new_syscall_list


def generate_next_syscall(sequence):
    input_ids_tensor = tokenizer.tokenize_sequence(sequence, return_tensors="pt")
    input_ids = input_ids_tensor.data['input_ids']
    mask_token_index = torch.where(input_ids == 211693)[1]
    mask_token_logits = mask_model(input_ids).logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    syscalls = []
    for token in top_5_tokens:
        syscalls.append(tokenizer.decode([token]))
    return syscalls


app = Flask(__name__)


@app.route('/', methods=['POST'])
def handle_post_request():
    syscall_json = request.get_json()
    print("syscallsData: ", syscall_json)

    syscall_list = []
    for key, value in syscall_json.items():
        if key == "Syscalls":
            syscall_list = value
    syscall_list = validate_syscall(syscall_list)

    sequence = [CLS] + syscall_list + [SEP]
    print("sequence: ", sequence, "\n")
    next_syscalls = generate_next_syscall(sequence)

    print("next_syscalls: ", next_syscalls, "\n")
    response = {'State': 0, 'Syscall': next_syscalls[0]}
    return jsonify(response)


if __name__ == '__main__':
    print("launch syzLLM server...")
    init_env()
    app.run(host='0.0.0.0', port=6678)
