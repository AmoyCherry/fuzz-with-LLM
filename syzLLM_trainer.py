import torch
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaConfig, DistilBertForMaskedLM, DistilBertConfig
from pathlib import Path

from syz_tokenizer import SyzTokenizer
from utils import ModelPath


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


def get_dataloader(syzTokenizer: SyzTokenizer):
    encodings = get_encodings_from_tokenfile(syzTokenizer)
    dataset = Dataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    return loader


def get_encodings_from_tokenfile(syzTokenizer: SyzTokenizer):
    batch = []
    for i in Path("./tokens/").glob('**/*.txt'):
        batch += syzTokenizer.get_sequence_batch(i)
    print("sequences size: ", len(batch))
    labels = torch.tensor([x.input_ids for x in batch])
    mask = torch.tensor([x.attention_mask for x in batch])

    # make copy of labels tensor, this will be input_ids
    input_ids = labels.detach().clone()
    # create random array of floats with equal dims to input_ids
    rand = torch.rand(input_ids.shape)
    # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
    mask_arr = (rand < .15) * (input_ids != syzTokenizer.tokenizer.pad_token_id) * (input_ids != syzTokenizer.tokenizer.cls_token_id) * (input_ids != syzTokenizer.tokenizer.sep_token_id)
    # loop through each row in input_ids tensor (cannot do in parallel)
    for j in range(input_ids.shape[0]):
        # get indices of mask positions from mask array
        selection = torch.flatten(mask_arr[j].nonzero()).tolist()
        temp = input_ids[j, selection]
        # mask input_ids
        input_ids[j, selection] = syzTokenizer.tokenizer.mask_token_id

    print(input_ids.shape)

    encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
    return encodings


DISTILBERT = "distilbert"
BERT = "bert"


class SyzLLMTrainer:
    def __init__(self, model=DISTILBERT):
        self.tokenizer = SyzTokenizer()

        config = RobertaConfig(
            vocab_size=self.tokenizer.vocab_size(),  # we align this to the tokenizer vocab_size
            max_position_embeddings=256,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )
        distil_config = DistilBertConfig(
            vocab_size=self.tokenizer.vocab_size(),
            max_position_embeddings=256,
            dropout=0.2,
        )

        print("Using model: ", model)
        if model == DISTILBERT:
            self.model = DistilBertForMaskedLM(distil_config)
        else:
            self.model = RobertaForMaskedLM(config)

        self.device = None

    def setup_device(self):
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        print(f"Using device: {self.device}")
        # and move our model over to the selected device
        self.model.to(self.device)

    def train(self):
        self.setup_device()
        # activate training mode
        self.model.train()
        # initialize optimizer
        optim = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        epochs = 2

        loader = get_dataloader(self.tokenizer)
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
    SyzLLMTrainer().train()
