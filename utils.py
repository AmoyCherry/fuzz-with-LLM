import json

TokenizerPath = "./SyzTokenizer"
GPT2TokenizerPath = "./GPT2Tokenizer"
DummyVocabFilePath = "./DummySyzTokenizer/vocab.txt"
VocabFilePath = "./vocab/vocab.txt"
SyzTokenizerVocabFilePath = "./SyzTokenizer/vocab.txt"
ModelPath = "SyzLLM"
ConfigPath = "./SyzLLM_training_config.json"

CLS = "[CLS]"
SEP = "[SEP]"
UNK = "[UNK]"

UNK_idx = 142830
MASK_idx = 142831

with open(ConfigPath) as config_file:
    config = json.load(config_file)

# Accessing configuration for later use
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
LEARNING_RATE = config["training"]["learning_rate"]
VALIDATION_SPLIT_PERCENTAGE = config["training"]["validation_split_percentage"]

NUM_WORKERS = config["dataloader"]["num_workers"]
PREFETCH_FACTOR = config["dataloader"]["prefetch_factor"]

MAX_POSITION_EMBEDDINGS = config["model"]["max_position_embeddings"]
DROPOUT = config["model"]["dropout"]
ATTENTION_DROPOUT = config["model"]["attention_dropout"]
QA_DROPOUT = config["model"]["qa_dropout"]


def format_tokens(sequence: list[str]):
    if len(sequence) > 0 and sequence[-1] == '':
        return sequence[0:-1]
    else:
        return sequence
