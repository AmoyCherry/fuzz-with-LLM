import torch
from tokenizers.implementations import BertWordPieceTokenizer
from transformers import BertTokenizerFast
from pathlib import Path
from transformers import BertTokenizer

ModelPath = "tokenizer"

# select a old_tokenizer by set 'Tokenizer_str'
BertTokenizerFast_str = "BertTokenizerFast"
Tokenizer_str = BertTokenizerFast_str

TokenFilePath = [str(x) for x in Path("./tokens/").glob('**/*.txt')]


def batch_iterator():
    for i in Path("./tokens/").glob('**/*.txt'):
        file = open(i)
        yield file.read().splitlines()


class SyzTokenizerTrainer:
    def __init__(self, _tokenizer):
        self.tokenizer_str = _tokenizer

    def train(self):
        if self.tokenizer_str == BertTokenizerFast_str:
            bert = BertTokenizerFast('vocab.txt')
            bert.save_pretrained(ModelPath)


class SyzTokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(ModelPath)

    def tokenize_word(self, word: str):
        return self.tokenizer.convert_tokens_to_ids(word)

    def tokenize_sequence(self, sequence: list[str]) -> list:
        return self.tokenizer.encode_plus(sequence, is_split_into_words=False)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}


if __name__ == "__main__":
    if Path("tokenizer").exists() is False:
        tokenTrainer = SyzTokenizerTrainer(Tokenizer_str)
        tokenTrainer.train()

    syzTokenizer = SyzTokenizer()
    word1 = "mq_unlink(&(0x7f0000000000)='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\x00')"
    word2 = "write(r0, &(0x7f0000000020)=\"010000000000000000\", 0x9)"
    print(syzTokenizer.tokenize_sequence([word1, word2]))

    for i in Path("./tokens/").glob('**/*.txt'):
        with open(i) as file:
            lines = file.read().split('\n')
            batch = syzTokenizer.tokenize_sequence(lines)
            print("tokens size: %d", len(batch))

            labels = torch.tensor([x.input_ids for x in batch])
            mask = torch.tensor([x.attention_mask for x in batch])
            # make copy of labels tensor, this will be input_ids
            input_ids = labels.detach().clone()
            # create random array of floats with equal dims to input_ids
            rand = torch.rand(input_ids.shape)
            # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
            mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
            # loop through each row in input_ids tensor (cannot do in parallel)
            for j in range(input_ids.shape[0]):
                # get indices of mask positions from mask array
                selection = torch.flatten(mask_arr[j].nonzero()).tolist()
                # mask input_ids
                input_ids[j, selection] = 3  # our custom [MASK] token == 3

            print(input_ids.shape)

            encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
            dataset = Dataset(encodings)
            loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
