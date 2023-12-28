TokenizerPath = "SyzTokenizer"
VocabFilePath = "vocab/vocab.txt"
ModelPath = "SyzLLM"


def format_tokens(sequence: list[str]):
    if len(sequence) > 0 and sequence[-1] == '':
        return sequence[0:-1]
    else:
        return sequence
