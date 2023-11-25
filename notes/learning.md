> material:
>
> [AFL brief](https://paper.seebug.org/841/)
>
> [a simple intro](https://meetingcpp.com/mcpp/slides/2018/Structured%20fuzzing.pdf) 
>
> [fuzzing documents from google](https://github.com/google/fuzzing)



## Efficient Ways

[A entry-level guid: go beyond coverage-guided fuzzing](https://i.blackhat.com/USA-19/Wednesday/us-19-Metzman-Going-Beyond-Coverage-Guided-Fuzzing-With-Structured-Fuzzing.pdf)

![image-20231008220701992](/Users/wangwenzhi/Library/Application Support/typora-user-images/image-20231008220701992.png)

### coverage-guided mutation-based fuzzers

- AFL
- lib-fuzzer

### structure-aware

> [examples](https://github.com/google/fuzzing/blob/master/docs/structure-aware-fuzzing.md)

Example 1:

```c
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  uint8_t Uncompressed[100];
  size_t UncompressedLen = sizeof(Uncompressed);
  if (Z_OK != uncompress(Uncompressed, &UncompressedLen, Data, Size))
    return 0;
  if (UncompressedLen < 2) return 0;
  if (Uncompressed[0] == 'F' && Uncompressed[1] == 'U') // legal uncompressed data
    abort();  // Boom
  return 0;
}
```

For the above code, it will rarely go through `abort()` which could make a deeper call stack and a higher coverage if using the default mutator that operate the compressed data. The process will destroy the data structure and cause a illegal uncompressed data.

So we can understand the logic and make a clever (structure-aware) mutator like below.

```c
extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size,
                                          size_t MaxSize, unsigned int Seed) {
  uint8_t Uncompressed[100];
  size_t UncompressedLen = sizeof(Uncompressed);
  size_t CompressedLen = MaxSize;
  if (Z_OK != uncompress(Uncompressed, &UncompressedLen, Data, Size)) {
    // The data didn't uncompress. Return a dummy...
  }
  UncompressedLen =
      LLVMFuzzerMutate(Uncompressed, UncompressedLen, sizeof(Uncompressed));
  if (Z_OK != compress(Data, &CompressedLen, Uncompressed, UncompressedLen))
    return 0;
  return CompressedLen;
}
```

One thing we need to pay attention to, which is that the mutator is operate on uncompressed data and then compress the mutated uncompressed data to ensure the return value is in the legal format.



Example 2:

To fuzz a database engine, for instance SQLite, we need wo generate and mutate SQLs. It's not very effective if gen meaningless string. So we can mutate fields and values and combine them in a legal format to make the SQLs can be excuted.



***So the term of structure-aware can be seen as a customized mutator which understand the target logic and can make the mutated value meet the type constraints and the excution flow pass through as much as possible if-else to reach deep code.*** 

***It is very suit for fuzzing with highly structured inputs.*** 