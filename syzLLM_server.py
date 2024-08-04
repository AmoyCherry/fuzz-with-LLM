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
    TEMPERATURE = 'temperature'
    TOP_K = 'top_k'
    TOP_P = 'top_p'
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

    if sampling_method == SamplingMethod.TEMPERATURE:
        top_tokens = sample_with_temperature(mask_token_logits, temperature)
    elif sampling_method == SamplingMethod.TOP_K:
        top_tokens = sample_with_top_k(mask_token_logits, top_k)
    elif sampling_method == SamplingMethod.TOP_P:
        top_tokens = sample_with_top_p(mask_token_logits, top_p)
    elif sampling_method == SamplingMethod.BEAM_SEARCH:
        top_tokens = beam_search_with_diversity(mask_token_logits, beam_width, diversity_penalty)
    else:
        raise ValueError("Invalid sampling method specified.")

    syscalls = []
    for token in top_tokens:
        call = tokenizer.decode([token])
        if "image" in call:
            continue
        syscalls.append(call)

    return syscalls


def sample_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(1)


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


def beam_search_with_diversity(logits, beam_width=5, diversity_penalty=1.0):
    batch_size, vocab_size = logits.size()
    scores = torch.zeros(batch_size, beam_width).to(logits.device)
    sequences = torch.zeros(batch_size, beam_width, 1).long().to(logits.device)

    # Initialize with the logits for the first step
    logits = logits.unsqueeze(1).expand(-1, beam_width, -1)  # [batch_size, beam_width, vocab_size]
    scores[:, 0] = logits[:, 0, :].topk(beam_width, dim=-1)[0].mean(dim=-1)

    for step in range(1):  # Assuming single-step generation for the masked token
        next_scores = scores.unsqueeze(-1) + logits  # [batch_size, beam_width, vocab_size]

        # Apply diversity penalty
        for i in range(beam_width):
            next_scores[:, i, :] -= diversity_penalty * step

        next_scores = next_scores.view(batch_size, -1)  # [batch_size, beam_width * vocab_size]
        top_scores, top_indices = next_scores.topk(beam_width, dim=-1)

        # Calculate indices for sequences
        beam_indices = top_indices // vocab_size  # Which beam each token came from
        token_indices = top_indices % vocab_size  # Which token was selected

        # Gather sequences
        sequences = sequences.gather(1, beam_indices.unsqueeze(-1).expand(-1, -1, sequences.size(-1)))
        sequences = torch.cat([sequences, token_indices.unsqueeze(-1)], dim=-1)

        scores = top_scores

    # Return the final sequences
    return sequences[:, :, 1].view(-1)  # Remove the initial placeholder and flatten


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
    #next_syscalls = fill_mask(sequence, sampling_method=SamplingMethod.TEMPERATURE, temperature=0.7)
    #next_syscalls = fill_mask(sequence, sampling_method=SamplingMethod.TOP_K, top_k=50)
    next_syscalls = fill_mask(sequence, sampling_method=SamplingMethod.TEMPERATURE, top_p=0.9)
    #next_syscalls = fill_mask(sequence, sampling_method=SamplingMethod.BEAM_SEARCH, beam_width=5, diversity_penalty=1.0)

    print("next_syscalls: ", next_syscalls, "\n")
    #idx = pick(len(syscall_list))
    #response = {'State': 0, 'Syscall': next_syscalls[idx]}
    response = {'State': 0, 'Syscall': next_syscalls}
    return jsonify(response)


if __name__ == '__main__':
    print("launch syzLLM server...")
    init_env()
    app.run(host='0.0.0.0', port=6678)
