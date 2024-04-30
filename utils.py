TokenizerPath = "./SyzTokenizer"
GPT2TokenizerPath = "./GPT2Tokenizer"
DummyVocabFilePath = "./DummySyzTokenizer/vocab.txt"
VocabFilePath = "./vocab/vocab.txt"
SyzTokenizerVocabFilePath = "./SyzTokenizer/vocab.txt"
ModelPath = "SyzLLM"

CLS = "[CLS]"
SEP = "[SEP]"
UNK = "[UNK]"

UNK_idx = 142830
MASK_idx = 142831


def format_tokens(sequence: list[str]):
    if len(sequence) > 0 and sequence[-1] == '':
        return sequence[0:-1]
    else:
        return sequence
