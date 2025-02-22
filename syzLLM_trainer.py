import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaConfig, DistilBertForMaskedLM, DistilBertConfig
from pathlib import Path

from syz_tokenizer import SyzTokenizer
from utils import ModelPath, BATCH_SIZE, NUM_WORKERS, PREFETCH_FACTOR, EPOCHS, LEARNING_RATE, \
    VALIDATION_SPLIT_PERCENTAGE, DROPOUT, ATTENTION_DROPOUT, QA_DROPOUT, \
    Distil_MAX_POSITION_EMBEDDINGS, BERT_MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE, NUM_ATTENTION_HEADS, NUM_HIDDEN_LAYERS, \
    TYPE_VOCAB_SIZE, SELECTEDMODEL, BERT


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
    full_dataset = Dataset(encodings)
    # Calculate the number of samples to include in each set
    train_size = int((1 - VALIDATION_SPLIT_PERCENTAGE) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Randomly split the dataset into training and validation datasets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create the training and validation dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR
    )

    return train_loader, val_loader


def get_encodings_from_tokenfile(syzTokenizer: SyzTokenizer):
    batch = []
    for i in Path("./tokens/").glob('**/*.txt'):
        batch += syzTokenizer.get_sequence_batch(i)
    print("sequences size: ", len(batch))

    labels = torch.tensor([x.input_ids for x in batch])
    attention_mask = torch.tensor([x.attention_mask for x in batch])

    input_ids = labels.detach().clone()
    print(input_ids.shape)
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < .15) * (input_ids != syzTokenizer.tokenizer.pad_token_id) * \
               (input_ids != syzTokenizer.tokenizer.cls_token_id) * \
               (input_ids != syzTokenizer.tokenizer.sep_token_id)

    labels = torch.full_like(input_ids, -100)

    for j in range(input_ids.shape[0]):
        selection = torch.flatten(mask_arr[j].nonzero()).tolist()
        if selection:
            original_tokens = input_ids[j, selection]
            input_ids[j, selection] = syzTokenizer.tokenizer.mask_token_id
            labels[j, selection] = original_tokens

    encodings = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    return encodings


class SyzLLMTrainer:
    def __init__(self):
        self.tokenizer = SyzTokenizer()

        bert_config = RobertaConfig(
            vocab_size=self.tokenizer.vocab_size(),
            max_position_embeddings=BERT_MAX_POSITION_EMBEDDINGS,
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_ATTENTION_HEADS,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            type_vocab_size=TYPE_VOCAB_SIZE
        )

        distilbert_config = DistilBertConfig(
            vocab_size=self.tokenizer.vocab_size(),
            max_position_embeddings=Distil_MAX_POSITION_EMBEDDINGS,
            dropout=DROPOUT,
            attention_dropout=ATTENTION_DROPOUT,
            qa_dropout=QA_DROPOUT
        )

        print("selected model: ", SELECTEDMODEL)
        if SELECTEDMODEL == BERT:
            self.model = RobertaForMaskedLM(bert_config)
        else:
            self.model = DistilBertForMaskedLM(distilbert_config)

        self.device = None

    def setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        # move our model over to the selected device
        self.model.to(self.device)

    def validate(self, validation_loader, writer):
        self.model.eval()  # set model to evaluation mode
        validation_loss = 0.0

        global_step = 0

        with torch.no_grad():
            loop = tqdm(validation_loader, leave=True)
            for batch in loop:
                # Same process as training to get inputs and labels
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loop.set_description('Validation ')
                loop.set_postfix(loss=loss.item())
                validation_loss += loss.item()
                writer.add_scalar('Validation/loss', loss.item(), global_step)
                global_step += 1

        if len(validation_loader) != 0:
            validation_loss /= len(validation_loader)
        return validation_loss

    def train(self, train_loader, validation_loader):
        self.setup_device()
        self.model.train()

        optim = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)

        best_validation_loss = float('inf')  # Keep track of the best validation loss

        epochs = EPOCHS

        for epoch in range(epochs):
            # setup loop with TQDM and dataloader
            writer = SummaryWriter()
            global_step = 0

            loop = tqdm(train_loader, leave=True)
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
                writer.add_scalar('Train/loss', loss.item(), global_step)
                global_step += 1  # Increment the global step

            self.model.save_pretrained(ModelPath + f"_{epoch}")

            # Validation Loop
            if len(validation_loader) == 0:
                validation_loss = self.validate(validation_loader, writer)
                print(f"Validation loss for epoch {epoch}: {validation_loss}")

        print('training done')

if __name__ == "__main__":
    syzLLM_trainer = SyzLLMTrainer()
    train_loader, validation_loader = get_dataloader(syzLLM_trainer.tokenizer)
    syzLLM_trainer.train(train_loader, validation_loader)
