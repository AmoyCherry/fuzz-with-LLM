import datetime
import difflib
import queue
import random
import re
import threading
import time

from flask import Flask, request, jsonify
from rapidfuzz import process, fuzz
from transformers import AutoModelForMaskedLM
import torch
import torch.nn.functional as F
from enum import Enum

from syz_tokenizer import SyzTokenizer
from utils import ModelPath, VocabFilePath, CLS, SEP, UNK_idx, UNK, SyzTokenizerVocabFilePath, ServerLogPath

tokenizer = SyzTokenizer()
mask_model = AutoModelForMaskedLM.from_pretrained(ModelPath)
syscall_dict = {}
syscall_name_dict = {}


class LogRecord:
    def __init__(self, t_num, c_num):
        self.tokenize_num = t_num
        self.call_num = c_num


log_queue = queue.Queue()


def log_worker(log_queue):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    while True:
        if log_queue.empty():
            time.sleep(1.0)
            continue
        record = log_queue.get()
        with open(ServerLogPath + f"{timestamp}.txt", 'a') as f:
            f.write(f"{record.tokenize_num} {record.call_num}\n")


def extract_syscall_name(syscall):
    description_pattern = re.compile(r'\b([a-zA-Z0-9_]+)\$')
    brackets_pattern = re.compile(r'\b([a-zA-Z0-9_]+)\(')
    match = description_pattern.search(syscall)
    if match:
        return match.group(1)[:syscall.index('$')]

    match = brackets_pattern.search(syscall)
    if match:
        return match.group(1)[:syscall.index('(')]

    return syscall


def init_env():
    names_path = "./names.txt"
    with open(names_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(' ', 1)
            syscall_name_dict[key] = value

    with open(SyzTokenizerVocabFilePath, 'r') as file:
        for line in file:
            syscall = line.strip()
            syscall_name = extract_syscall_name(syscall)
            syscall_dict.setdefault(syscall_name, list()).append(syscall)

    # thread = threading.Thread(target=log_worker, args=(log_queue,))
    # thread.start()
    # print("Log thread start...")


def find_most_similar(reference_string, strings_set):
    result = process.extractOne(reference_string, strings_set, scorer=fuzz.WRatio)
    return result[0] if result else None


async def validate_syscall(syscall_list):
    new_syscall_list = []
    tokenize_num = 0
    for syscall in syscall_list:
        syscall = replace_description_with_syzllm(syscall)
        syscall_name = extract_syscall_name(syscall)
        if tokenizer.tokenize_word(syscall) != UNK_idx:
            tokenize_num += 1
            new_syscall_list.append(syscall)
        elif syscall_name in syscall_dict:
            syscall_set = syscall_dict[syscall_name]
            similar_call = find_most_similar(syscall, syscall_set)
            if similar_call is not None:
                new_syscall_list.append(similar_call)
            else:
                idx = random.randint(0, len(syscall_set) - 1)
                new_syscall_list.append(syscall_set[idx])
        else:
            new_syscall_list.append(UNK)

    #log_queue.put(LogRecord(tokenize_num, len(syscall_list)))
    return new_syscall_list


def extract_call_name_in_resource(input):
    resource_pattern = r'@RSTART@((?:(?!@RSTART@).)*?)\$SyzLLM'
    return re.findall(resource_pattern, input)

def replace_description_with_syzllm(syscall):
    name_description_pattern = r'\$(.*?)\('
    syzllm_pattern = r'$SyzLLM('
    return re.sub(name_description_pattern, syzllm_pattern, syscall)


async def remove_syzllm_from_description(syscall):
    name_and_description_pattern = r'\b(.*?)\('

    call_replacement = syscall_name_dict[extract_syscall_name(syscall)] + '('
    replaced_call = re.sub(name_and_description_pattern, call_replacement, syscall, count=1)

    resources = extract_call_name_in_resource(replaced_call)
    for resource in resources:
        resource_replacement = syscall_name_dict[resource]
        replaced_call = re.sub(resource + r'\$SyzLLM', resource_replacement, replaced_call, count=1)

    return replaced_call

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
    #BEAM_SEARCH = 'beam_search'


async def fill_mask(sequence,
                    sampling_method=SamplingMethod.TOP_K,
                    temperature=1.0, top_k=25,
                    top_p=0.9,
                    beam_width=5, diversity_penalty=1.0):
    input_ids_tensor = tokenizer.tokenize_sequence(sequence, return_tensors="pt", max_length_arg=max(128, highest_power_of_2(len(sequence) + 2)*2))
    input_ids = input_ids_tensor.data['input_ids']
    mask_token_index = torch.where(input_ids == 208925)[1]
    model = mask_model
    mask_token_logits = model(input_ids).logits[0, mask_token_index, :]
    top_tokens = sample(mask_token_logits, sampling_method, temperature, top_k, top_p, beam_width, diversity_penalty)

    syscalls = []
    for token in top_tokens:
        call = tokenizer.decode([token])
        call = await remove_syzllm_from_description(call)
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
    #elif sampling_method == SamplingMethod.BEAM_SEARCH:
    #    return beam_search_one_step_with_diversity(logits, beam_width, diversity_penalty)
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


def sample_with_top_k(logits, k=25):
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

# NOTE! beam search seems not good enough
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


class CircularQueue:
    def __init__(self, capacity=5):
        self.queue = [0] * capacity
        self.capacity = capacity
        self.head = 0
        self.tail = -1
        self.size = 0

    def enqueue(self, value):
        if self.size < self.capacity:
            self.tail = (self.tail + 1) % self.capacity
            self.queue[self.tail] = value
            self.size += 1
        else:
            self.tail = (self.tail + 1) % self.capacity
            self.head = (self.head + 1) % self.capacity
            self.queue[self.tail] = value

    def get_sum(self):
        return sum(self.queue)


class SamplingMethodSelector:
    def __init__(self, initial_sampling=SamplingMethod.SAMPLE_TOP_P, initial_cover_sum=3000):
        self.covers_dict = {}
        for method in SamplingMethod:
            self.covers_dict[method] = CircularQueue()
            if method == SamplingMethod.TOP_K:
                self.covers_dict[method].enqueue(500)
            else:
                self.covers_dict[method].enqueue(initial_cover_sum)
        self.sorted_algorithms = []
        self.previous_coverage = 0
        self.current_sampling = initial_sampling

    def update_cover(self, coverage):
        if request_counter.has_request_per_cover() is False:
            return

        diff = coverage - self.previous_coverage
        self.previous_coverage = coverage

        algorithm = self.current_sampling
        if algorithm not in self.covers_dict:
            self.covers_dict[algorithm] = CircularQueue()
        queue = self.covers_dict[algorithm]
        queue.enqueue(diff)

        self.try_update_current_sampling()

    def try_update_current_sampling(self):
        if request_counter.should_reset():
            self.update_current_sampling()

    def update_current_sampling(self):
        self.update_sorted_algorithms()
        if self.sorted_algorithms:
            highest_sum_algorithm, highest_sum = self.sorted_algorithms[0]
            if highest_sum_algorithm == SamplingMethod.TOP_K and self.current_sampling == SamplingMethod.TOP_K and highest_sum <= 600:
                self.select_random_method()
            else:
                self.current_sampling = highest_sum_algorithm

    def update_sorted_algorithms(self):
        sum_scores = [(algo, self.covers_dict[algo].get_sum()) for algo in self.covers_dict]
        self.sorted_algorithms = sorted(sum_scores, key=lambda x: x[1], reverse=True)

    def select_random_method(self, exclude=SamplingMethod.TOP_K):
        choices = [method for method in SamplingMethod if method != exclude]
        self.current_sampling = random.choice(choices)


class RequestCounter:
    def __init__(self):
        self.request_counter = 0
        self.request_counter_per_cover = 0

    def count(self):
        self.request_counter += 1
        self.request_counter_per_cover += 1

    def has_request_per_cover(self):
        if self.request_counter_per_cover > 0:
            self.reset_per_cover()
            return True
        else:
            return False

    def reset_per_cover(self):
        self.request_counter_per_cover = 0

    def reset(self):
        self.request_counter = 0

    def should_reset(self):
        if self.request_counter > 75:
            self.reset()
            return True
        else:
            return False


app = Flask(__name__)

sample_method_selector = SamplingMethodSelector()
request_counter = RequestCounter()


@app.route('/cover', methods=['POST'])
def handle_cover():
    try:
        cover = int(request.data.decode("utf-8"))
        sample_method_selector.update_cover(cover)
        return "", 200
    except ValueError:
        return "wrong cover!", 400


@app.route('/', methods=['POST'])
async def handle_post_request():
    request_counter.count()

    syscall_json = request.get_json()

    syscall_list = []
    for key, value in syscall_json.items():
        if key == "Syscalls":
            syscall_list = value
    syscall_list = await validate_syscall(syscall_list)

    #sequence = [CLS] + syscall_list + [SEP]
    sequence = syscall_list
    next_syscalls = await fill_mask(sequence, sampling_method=sample_method_selector.current_sampling, temperature=0.7)
    # idx = pick(len(syscall_list))
    # response = {'State': 0, 'Syscall': next_syscalls[idx]}

    response = {'State': 1, 'Syscall': ''}

    if len(next_syscalls) > 0:
        print(f"samping: {sample_method_selector.current_sampling.value}\nnext_syscalls: {next_syscalls[0]}\n")
        response['State'] = 0
        response['Syscall'] = next_syscalls[0]

    return jsonify(response)


if __name__ == '__main__':
    print("launch syzLLM server...")
    init_env()
    app.run(host='0.0.0.0', port=6678)
