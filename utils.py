import json

TokenizerPath = "./SyzTokenizer"
GPT2TokenizerPath = "./GPT2Tokenizer"
DummyVocabFilePath = "./DummySyzTokenizer/vocab.txt"
VocabFilePath = "./vocab/vocab.txt"
SyzTokenizerVocabFilePath = "./SyzTokenizer/vocab.txt"
ModelPath = "SyzLLM"
ConfigPath = "./SyzLLM_training_config.json"
ServerLogPath = "./log/"

CLS = "[CLS]"
SEP = "[SEP]"
UNK = "[UNK]"

UNK_idx = 143064
MASK_idx = 143065

with open(ConfigPath) as config_file:
    config = json.load(config_file)

# Accessing configuration for later use
# training
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
LEARNING_RATE = config["training"]["learning_rate"]
VALIDATION_SPLIT_PERCENTAGE = config["training"]["validation_split_percentage"]

SELECTEDMODEL = config["selectedModel"]
# model.bert
HIDDEN_SIZE = config["model"]["bert"]["hidden_size"]
NUM_ATTENTION_HEADS = config["model"]["bert"]["num_attention_heads"]
NUM_HIDDEN_LAYERS = config["model"]["bert"]["num_hidden_layers"]
TYPE_VOCAB_SIZE = config["model"]["bert"]["type_vocab_size"]
BERT_MAX_POSITION_EMBEDDINGS = config["model"]["bert"]["max_position_embeddings"]
# model.distilbert
Distil_MAX_POSITION_EMBEDDINGS = config["model"]["distilbert"]["max_position_embeddings"]
DROPOUT = config["model"]["distilbert"]["dropout"]
ATTENTION_DROPOUT = config["model"]["distilbert"]["attention_dropout"]
QA_DROPOUT = config["model"]["distilbert"]["qa_dropout"]

# dataloader
NUM_WORKERS = config["dataloader"]["num_workers"]
PREFETCH_FACTOR = config["dataloader"]["prefetch_factor"]

BERT = "bert"
DISTILBERT = "distilbert"

def format_tokens(sequence: list[str]):
    if len(sequence) > 0 and sequence[-1] == '':
        return sequence[0:-1]
    else:
        return sequence
