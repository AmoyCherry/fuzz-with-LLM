TokenizerPath = "./SyzTokenizer"
GPT2TokenizerPath = "./GPT2Tokenizer"
DummyVocabFilePath = "./DummySyzTokenizer/vocab.txt"
VocabFilePath = "./SyzTokenizer/vocab.txt"
ModelPath = "SyzLLM"

CLS = "[CLS]"
SEP = "[SEP]"
UNK = "[UNK]"

UNK_idx = 116356
MASK_idx = 116357


def format_tokens(sequence: list[str]):
    if len(sequence) > 0 and sequence[-1] == '':
        return sequence[0:-1]
    else:
        return sequence
