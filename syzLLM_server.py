import random

from flask import Flask, request, jsonify
from transformers import AutoModelForMaskedLM
import torch
import torch.nn.functional as F
from enum import Enum

from syz_tokenizer import SyzTokenizer
from utils import ModelPath, VocabFilePath, CLS, SEP, UNK_idx, UNK, SyzTokenizerVocabFilePath

tokenizer = SyzTokenizer()
mask_model = AutoModelForMaskedLM.from_pretrained(ModelPath)
syscall_name_dict = {}


def extract_syscall_name(syscall):
    if '(' in syscall:
        return syscall[:syscall.index('(')]

    return syscall


def init_env():
    with open(SyzTokenizerVocabFilePath, 'r') as file:
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


def highest_power_of_2(N):
    # if N is a power of two simply return it
    if (not (N & (N - 1))):
        return N

    # else set only the most significant bit
    return 0x8000000000000000 >> (64 - N.bit_length())


class SamplingMethod(Enum):
    TOP_K = 'top_k'
    TEMPERATURE = 'temperature'
    SAMPLE_TOP_K = 'sample_top_k'
    SAMPLE_TOP_P = 'sample_top_p'
    BEAM_SEARCH = 'beam_search'


def fill_mask(sequence,
              sampling_method=SamplingMethod.TOP_K,
              temperature=1.0, top_k=50,
              top_p=0.9,
              beam_width=5, diversity_penalty=1.0):
    input_ids_tensor = tokenizer.tokenize_sequence(sequence, return_tensors="pt", max_length_arg=max(128, highest_power_of_2(len(sequence) + 2) * 2))
    input_ids = input_ids_tensor.data['input_ids']
    mask_token_index = torch.where(input_ids == 182605)[1]
    mask_token_logits = mask_model(input_ids).logits[0, mask_token_index, :]
    top_tokens = sample(mask_token_logits, sampling_method, temperature, top_k, top_p, beam_width, diversity_penalty)

    syscalls = []
    for token in top_tokens:
        call = tokenizer.decode([token])
        if "image" in call:
            continue
        syscalls.append(call)

    return syscalls


def sample(logits, sampling_method, temperature=1.0, k=15, top_p=0.9, beam_width=5, diversity_penalty=1.0):
    if sampling_method == SamplingMethod.TEMPERATURE:
        return sample_with_temperature(logits, temperature)
    elif sampling_method == SamplingMethod.SAMPLE_TOP_K:
        return sample_with_top_k(logits, k)
    elif sampling_method == SamplingMethod.SAMPLE_TOP_P:
        return sample_with_top_p(logits, top_p)
    elif sampling_method == SamplingMethod.BEAM_SEARCH:
        return beam_search_one_step_with_diversity(logits, beam_width, diversity_penalty)
    elif sampling_method == SamplingMethod.TOP_K:
        return top_k(logits)
    else:
        raise ValueError("Invalid sampling method specified.")


def sample_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(1)


def top_k(logits, k=6):
    return [torch.topk(logits, k, dim=1).indices[0].tolist()[pick(k)]]


def sample_with_top_k(logits, k=50):
    values, indices = torch.topk(logits, k=k)
    probs = F.softmax(values, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return indices.gather(-1, next_token).squeeze(1)


def sample_with_top_p(logits, top_p=0.9):
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Create a mask to remove tokens beyond the cutoff index
    sorted_indices_to_remove = cumulative_probs > top_p

    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter sorted indices to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

    # Set logits of tokens beyond the cutoff index to -inf
    logits[indices_to_remove] = -float('Inf')

    # Compute probabilities from logits and sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def beam_search_one_step_with_diversity(logits, k=10, diversity_penalty=1.0):
    """
    Perform one step of beam search with diversity.

    Parameters:
    - logits: Tensor of logits from which to select the top-k.
    - k: Number of top elements to select.
    - diversity_penalty: Penalty to apply to logits for encouraging diversity.

    Returns:
    - List of indices representing the top-k selections with diversity.
    """

    # Apply softmax to convert logits into probabilities for easier handling
    probs = F.softmax(logits, dim=-1).squeeze(0)

    # Initialize an empty list to hold the final indices with diversity
    final_indices = torch.empty(k, dtype=torch.long)

    for i in range(k):
        # Get the next top element
        _, index = torch.topk(probs, k=1)
        index = index.item()

        # Append the selected index to the final list
        final_indices[i] = index

        # Apply a penalty to the selected index to encourage diversity
        probs[index] -= diversity_penalty

        # Ensure the probability does not become negative
        probs[index] = max(probs[index], 0)

        # Re-normalize the probabilities
        if probs.sum() > 0:
            probs /= probs.sum()

    return [final_indices[pick(k)]]


def pick(n):
    probability = random.random()
    if probability < 0.5:
        return 0
    elif probability < 0.9:
        return min(1, n - 1)
    elif probability < 0.95:
        return min(2, n - 1)
    elif probability < 0.96:
        return min(3, n - 1)
    elif probability < 0.97:
        return min(4, n - 1)
    else:
        return min(5, n - 1)



app = Flask(__name__)


@app.route('/', methods=['POST'])
def handle_post_request():
    syscall_json = request.get_json()
    #print("syscallsData: ", syscall_json)

    syscall_list = []
    for key, value in syscall_json.items():
        if key == "Syscalls":
            syscall_list = value
    syscall_list = validate_syscall(syscall_list)

    #sequence = [CLS] + syscall_list + [SEP]
    sequence = syscall_list
    #print("sequence: ", sequence, "\n")
    next_syscalls = fill_mask(sequence, sampling_method=SamplingMethod.SAMPLE_TOP_P, temperature=0.7)

    print("next_syscalls: ", next_syscalls, "\n")
    #idx = pick(len(syscall_list))
    #response = {'State': 0, 'Syscall': next_syscalls[idx]}
    response = {'State': 0, 'Syscall': next_syscalls}
    return jsonify(response)


if __name__ == '__main__':
    print("launch syzLLM server...")
    init_env()
    app.run(host='0.0.0.0', port=6678)
