## 26-Otc-2024

### What is blocking us?

#### [Resolved✅] Evaluation

Syzkaller crashed after running 600k programs. Reported `no output from test machine`.

- cause: may caused by syscall dict having a large size;
- solution: minimize syscall dict;

## 25-Otc-2024

### Progress

1. Implemented [SyzLLM-Batch](https://github.com/AmoyCherry/fuzz-with-LLM/pull/7);

2. Minimize token size further;

- How: Remove all descriptions and assign const string for training; and retrieve descriptions before returning to the client;

- Result: 

  | token size | before (k) | now (k) | total calls (k) |
  | ---------- | ---------- | ------- | --------------- |
  | small set  | 280        | 200     | 1,500           |
  | large set  | 560        | 440     | 4,400           |

#### Evaluations

- SyzLLM-batch
- SyzLLM

### What is blocking us?

#### [Unresolved🛑] Design

- answer quality
  - The token size should be as small as possible than the total calls, this has a huge impact on the quality and diversity of the answers by influencing the average learning material per syscall.

### [Unfinished🚀] Ongoing

#### Engagement 5 data set from DAPAR

- value: may help us get a huge data set so that possibly get a small ratio that token size over total calls.

