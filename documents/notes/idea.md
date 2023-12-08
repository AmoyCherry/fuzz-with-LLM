

**Knowledge Transfer**

**Pre-trained distillation**



## Theory

Healer shows the impact of the relation learning-based CGF. But the transformer can be better than it in two aspects 1) training with position info of tokens 2) more accurate probability distribution.

> Compared to state-of-the-art kernel fuzzers such as **Syzkaller** and **Moonshine**, HEALER improves branch coverage by **28%** and **21%**, while achieving a speedup of **2.2×** and **1.8×**, respectively. In addition, HEALER detected 218 vulnerabilities, **33** of which are previously unknown and have been confirmed by the corresponding kernel maintainers.

mooshine transparent-computing

less parameters 

过拟合问题

收集细分模块(network, filesystem)的syscalls

## mutation strategy

Build a graph for extracted syscall relations from LLM.

#### Stage 1: sequance size 0 -> 1

Selected serveral syscalls as init sequences according to some algos. 

> E.g. select syscalls that have the highest dgree or smallest.

#### Stage 2: n -> n + 1

Refer to [healer](https://github.com/AmoyCherry/fuzz-with-LLM/blob/main/notes/reading.md#relation-table-guided-generation-and-mutation).

> N: syscall number

The time complexity of add a syscall to a sequence: 

$O(SubSequanceAVGSize*N)$ -> $O(·*AVGDegree)$

If it is a static graph (won't change in fuzzing process), it can be more faster by using PriorityQueue:

$O(SubSequanceAVGSize)$

> Considering a penalty and reward mechanism by leveraging coverage feedback. May it can be apply for seeds selection and input execution order.

