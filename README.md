# fuzz-with-LLM

> [reading notes](https://github.com/AmoyCherry/fuzz-with-LLM/blob/main/notes/reading.md)
>
> [idea](https://github.com/AmoyCherry/fuzz-with-LLM/blob/main/notes/idea.md)
>
> [research based on syzkaller (Organized by Google)](https://github.com/google/syzkaller/blob/master/docs/research.md)

## Goals

By **TRAINING** an LLM to Generate and systhesis high-quality seeds (syscall sequences) which better than manual-ruled ones from the original syzkaller.

### Questions

#### Q1: We can use LLM tools to get syscall relations. But there is a concern need to be addressed.

- Totally random sequences are not expected to get efficiency. And make-sense sequences can get a more higher coverage, but it seems like an 'expected behavior' so it may not be easy to find bugs as well. We should add some 'potencial malicious' syscalls into make-sence sequence so that get unexpected crashes. Or follow relation-learning idea from [healer](https://github.com/AmoyCherry/fuzz-with-LLM/blob/main/notes/reading.md#healer---sosp-2021).

- make-sense sequence for chatGPT:

  <img src="./documents/assets/gpt-1.png" alt="img" style="zoom: 50%;" />

- two examples to get relation

  <img src="./documents/assets/gpt-2.png" alt="img" style="zoom: 50%;" />

  <img src="./documents/assets/gpt-3.png" alt="img" style="zoom: 50%;" />

  

  > The knowledge base can be offline since it's already learned over. So that we can query the syscalls relation by scripts and draw a relation graph before fuzzing. ~~In addition, to incorperate realtime query, we can avoid some relation missing in pre-learned cache.~~



#### Q2: How to deal with the quality of seeds gen-ed by LLM?

- LLM can discover explicit relation between syscalls but not good at implicit ones. But the implicit realtions dont show a powerful influence from [RLTrace](https://github.com/AmoyCherry/fuzz-with-LLM/blob/main/notes/reading.md#rltrace-synthesizing-high-quality-system-call-traces-for-os-fuzz-testing---isc-2023).



#### Q3-1: Are the run time sessions with LLM supposed to be a bottleneck of fuzzing (need to be alleviate)? 

1). the hardware resources cost.  

2). the throughput (the delay time of making answers).



#### Q3-3: ~~Is it possible and neccesary to make benefits by leverage/combine LLM during the fuzzing process?~~ How to boot performance for run time session with LLM?



#### Q4: May there be some resources that could be training text before fuzzing?



Q5: We should choose an open-source model.

- LLaMa;
- 

