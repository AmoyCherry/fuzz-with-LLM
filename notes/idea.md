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

