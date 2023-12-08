## Train SyzLLM

> ref:
>
> [How to Train BERT from Scratch](https://thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python)
>
> [huggingface](https://huggingface.co/blog/pretraining-bert#3-preprocess-the-dataset)

### 1. Prepare dataset

### 1.1 Mooshine

#### build mooshine

NOTE:

1. set GOPATH at first. E.g. GOPATH = $HOME/gocode
2. For the error "can not find syzkaller/../ifuzz/generated", copy all files under ../ifuzz/x86/ to ../ifuzz.

#### run and get corpus.db

### 1.2 Syzkaller

Download corpus.db from [google drive](https://groups.google.com/g/syzkaller/c/77ObybwNnig?pli=1).

### 1.3 drapa (To be start)

https://github.com/darpa-i2o/Transparent-Computing

### 1.4 Parse corpus.db to split token files

A trace in corpus.db looks like below. It's consist of several syscalls with arguments (tokens) and we get the number of **120k** tokens now.

**The format of the tokens need a explanation.**

![token](../assets/token.png)

### 2. Train a tokenizer

### 3. Train the model