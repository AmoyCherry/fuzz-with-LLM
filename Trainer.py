import os
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast, RobertaForMaskedLM, RobertaConfig, pipeline
from pathlib import Path
from transformers import BertTokenizer

TokenizerPath = "SyzTokenizer"
VocabFilePath = "vocab/vocab.txt"
ModelPath = "SyzLLM"

SEPTokenID = 3298
PADTokenID = 3300
CLSTokenID = 3301
MASKTokenID = 3302

# select a old_tokenizer by set 'Tokenizer_str'
BertTokenizerFast_str = "BertTokenizerFast"
Tokenizer_str = BertTokenizerFast_str


def remove_symbols(sequence: list[str]):
    if len(sequence) > 0 and sequence[-1] == '':
        return sequence[0:-1]
    else:
        return sequence


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


def get_dataloader():
    encodings = get_encodings_from_tokenfile()
    dataset = Dataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    return loader


def get_encodings_from_tokenfile():
    batch = []
    for i in Path("./tokens/").glob('**/*.txt'):
        batch += syzTokenizer.get_sequence_batch(i)
    print("tokens size: ", len(batch))
    labels = torch.tensor([x.input_ids for x in batch])
    mask = torch.tensor([x.attention_mask for x in batch])

    # make copy of labels tensor, this will be input_ids
    input_ids = labels.detach().clone()
    # create random array of floats with equal dims to input_ids
    rand = torch.rand(input_ids.shape)
    # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
    mask_arr = (rand < .15) * (input_ids != PADTokenID) * (input_ids != CLSTokenID) * (input_ids != SEPTokenID)
    # loop through each row in input_ids tensor (cannot do in parallel)
    for j in range(input_ids.shape[0]):
        # get indices of mask positions from mask array
        selection = torch.flatten(mask_arr[j].nonzero()).tolist()
        # mask input_ids
        input_ids[j, selection] = MASKTokenID  # our custom [MASK] token == 3

    print(input_ids.shape)

    encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
    return encodings


class SyzTokenizerTrainer:
    def __init__(self, _tokenizer):
        self.tokenizer_str = _tokenizer

    def train(self):
        if self.tokenizer_str == BertTokenizerFast_str:
            dummy_bert = BertTokenizerFast(VocabFilePath)
            os.remove(VocabFilePath)
            dummy_bert.save_pretrained("vocab")
            print("extract vocabulary, ignore 'holes'")
            bert = BertTokenizerFast(VocabFilePath)
            bert.save_pretrained(TokenizerPath)


class SyzTokenizer:
    def __init__(self):
        if Path(TokenizerPath).exists() is False:
            tokenTrainer = SyzTokenizerTrainer(Tokenizer_str)
            tokenTrainer.train()
        self.tokenizer = BertTokenizer.from_pretrained(TokenizerPath)

    def vocab_size(self):
        return self.tokenizer.vocab_size

    def tokenize_word(self, word: str):
        return self.tokenizer.convert_tokens_to_ids(word)

    def tokenize_sequence(self, sequence: list[str]):
        return self.tokenizer.encode_plus(sequence, is_split_into_words=False, max_length=16, padding='max_length', truncation=True)

    def get_sequence_batch(self, filename):
        batch = []
        with open(filename) as file:
            sequences = file.read().split('[SEP]\n')
            for sequence in sequences:
                if sequence == '':
                    continue
                batch.append(self.tokenize_sequence(remove_symbols(sequence.split('\n'))))
        return batch


class SyzLLMTrainer:
    def __init__(self, _vocab_size):
        config = RobertaConfig(
            vocab_size=_vocab_size,  # we align this to the tokenizer vocab_size
            max_position_embeddings=514,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1
        )
        self.model = RobertaForMaskedLM(config)
        self.device = None

    def setup_device(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # and move our model over to the selected device
        self.model.to(self.device)

    def train(self):
        self.setup_device()
        # activate training mode
        self.model.train()
        # initialize optimizer
        optim = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        epochs = 2

        loader = get_dataloader()
        for epoch in range(epochs):
            # setup loop with TQDM and dataloader
            loop = tqdm(loader, leave=True)
            for batch in loop:
                # initialize calculated gradients (from prev step)
                optim.zero_grad()
                # pull all tensor batches required for training
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                # process
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                # extract loss
                loss = outputs.loss
                # calculate loss for every parameter that needs grad update
                loss.backward()
                # update parameters
                optim.step()
                # print relevant info to progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

        print('training done')
        self.model.save_pretrained(ModelPath)


if __name__ == "__main__":
    syzTokenizer = SyzTokenizer()
    if Path(ModelPath).exists() is False:
        # vocab_size = syzTokenizer.vocab_size()
        # vocab_size should match the size of tokenizer's vocabulary
        # print("SyzTokenizer has vocab_size: ", vocab_size)
        syzLLMTrainer = SyzLLMTrainer(3304)
        syzLLMTrainer.train()

    word1 = []
    word2 = []
    word3 = []
    sequence = [word1, word2, word3]

    fill = pipeline('fill-mask', model='SyzLLM', tokenizer='SyzTokenizer')
    fill(f'{fill.tokenizer.encode_plus(sequence, is_split_into_words=False, max_length=16, padding="max_length", truncation=True)} {fill.tokenizer.mask_token}')
