import os
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast, RobertaForMaskedLM, RobertaConfig, pipeline, top_k_top_p_filtering, AutoModelForMaskedLM
from pathlib import Path
from transformers import BertTokenizer
from torch.nn import functional as F


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
            print("extract vocabulary, ignore the above 'holes'")

            bert = BertTokenizerFast(VocabFilePath)
            bert.save_pretrained(TokenizerPath)
            print("save SyzTokenizer")


class SyzTokenizer:
    def __init__(self):
        if Path(TokenizerPath).exists() is False:
            tokenTrainer = SyzTokenizerTrainer(Tokenizer_str)
            tokenTrainer.train()
        self.tokenizer = BertTokenizer.from_pretrained(TokenizerPath)

    def mask_token(self):
        return self.tokenizer.mask_token()

    def decode(self, tokens: []):
        return self.tokenizer.decode(tokens)

    def decode_token(self, tokens: []):
        return self.tokenizer.decode(tokens)

    def mask_token_id(self):
        return self.tokenizer.mask_token_id()

    def vocab_size(self):
        return self.tokenizer.vocab_size

    def tokenize_word(self, word: str):
        return self.tokenizer.convert_tokens_to_ids(word)

    def tokenize_sequence(self, sequence: list[str], return_tensors=None):
        return self.tokenizer.encode_plus(sequence, is_split_into_words=False, max_length=16, padding='max_length', truncation=True, return_tensors=return_tensors)

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
    model = AutoModelForMaskedLM.from_pretrained(ModelPath)
    #model = AutoModelForCausalLM.from_pretrained(ModelPath)

    word1 = 'mmap(&(0x7f0000000000/0x1000)=nil, 0x1000, 0x3, 0x32, 0xffffffffffffffff, 0x0)'
    word2 = "open(&(0x7f0000000010)='fcntl215DIbFE\\x00', 0xc2, 0x180)"
    word3 = 'read(r0, &(0x7f000000001e)=\"\"/33, 0x21)'
    word4 = 'pipe(&(0x7f0000000008)={<r0=>0xffffffffffffffff})'
    word5 = "open(&(0x7f0000000000)='/lib/x86_64-linux-gnu/libc.so.6\\x00', 0x80000, 0x0)"
    MASK = "[MASK]"
    sequence1 = [word1, word2, word3]
    sequence2 = [word1, word2, word3, MASK]

    input_ids = syzTokenizer.tokenize_sequence(sequence1, return_tensors="pt")
    input_ids = input_ids.data['input_ids']
    # get logits of last hidden state
    next_token_logits = model(input_ids).logits[:, -1, :]

    # filter
    filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=5, top_p=1.0)

    # sample
    probs = F.softmax(filtered_next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    generated = torch.cat([input_ids, next_token], dim=-1)

    resulting_string = syzTokenizer.decode(generated.tolist()[0])
    print(resulting_string)

    input_ids = syzTokenizer.tokenize_sequence(sequence2, return_tensors="pt")
    input_ids = input_ids.data['input_ids']
    mask_token_index = torch.where(input_ids == 3301)[1]
    mask_token_logits = model(input_ids).logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    resulting_string = syzTokenizer.decode(top_5_tokens)
    print(resulting_string)
