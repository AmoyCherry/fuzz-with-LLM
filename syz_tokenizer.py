from pathlib import Path

from transformers import BertTokenizerFast, BertTokenizer

import utils


def train_tokenizer_from_bert():
    dummy = BertTokenizerFast(utils.DummyVocabFilePath)
    dummy.save_pretrained("vocab")
    print("extract vocabulary, ignore the above 'holes'")

    bert_tokenizer = BertTokenizerFast(utils.VocabFilePath)
    bert_tokenizer.save_pretrained(utils.TokenizerPath)
    print("save SyzTokenizerFromBert")


class SyzTokenizer:
    def __init__(self):
        if Path(utils.TokenizerPath).exists() is False:
            train_tokenizer_from_bert()
        self.tokenizer = BertTokenizer.from_pretrained(utils.TokenizerPath)
        self.deduplication_set = set()

    def tokenizer(self):
        return self.tokenizer

    def decode(self, tokens: []):
        return self.tokenizer.decode(tokens)

    def decode_token(self, tokens: []):
        return self.tokenizer.decode(tokens)

    def mask_token(self):
        return self.tokenizer.mask_token_id()

    def vocab_size(self):
        return self.tokenizer.vocab_size

    def tokenize_word(self, word: str):
        return self.tokenizer.convert_tokens_to_ids(word)

    def tokenize_sequence(self, sequence: list[str], return_tensors=None, max_length_arg=256):
        return self.tokenizer.encode_plus(sequence, is_split_into_words=False, max_length=max_length_arg, padding='max_length', truncation=True, return_tensors=return_tensors)


    def get_sequence_batch(self, filename):
        batch = []
        with open(filename) as file:
            sequences = file.read().split('[SEP]\n')
            for sequence in sequences:
                if sequence in self.deduplication_set:
                    continue
                self.deduplication_set.add(sequence)

                if sequence == '':
                    continue
                batch.append(self.tokenize_sequence(utils.format_tokens(sequence.split('\n'))))
        return batch


if __name__ == '__main__':
    train_tokenizer_from_bert()
