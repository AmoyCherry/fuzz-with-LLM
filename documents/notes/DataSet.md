Train two models based on different datasets, one uses all datasets and another one only use the traces collected from actual running apps.

## Dynamic malware detection and phylogeny analysis using process mining

> [Dataset](https://github.com/mlbresearch/syscall-traces-dataset)
>
> can be used for us? - YES✅

Small sequnce of syscalls with args collected from 1200 trusted and malicious applications across ten malware families on mobile phone.

We may need a filter for this dataset to adjust the massive number of `clock_gettime`. I'm concern about it will impact the model to predict clock_gettime frequently. 

```
#dataset snippest
0.000000 clock_gettime(CLOCK_MONOTONIC, {23721, 30555913}) = 0 <0.000007>
0.000136 epoll_wait(28, {{EPOLLIN, {u32=23, u64=23}}}, 16, -1) = 1 <0.020972>
0.021129 read(23, "W", 16)         = 1 <0.000010>
0.000041 clock_gettime(CLOCK_MONOTONIC, {23721, 51779004}) = 0 <0.000006>
0.000083 mprotect(0x9f5ff000, 8192, PROT_READ|PROT_WRITE) = 0 <0.000014>
0.000096 writev(3, [{"\x04", 1}, {"\x00", 1}, {"org.json.JSONException: End of i"..., 56}], 3) = 58 <0.000020>
```

## On Improving Deep Learning Trace Analysis with System Call Arguments

> [Dataset 1](https://zenodo.org/records/4091287#.X4hhGNjpNQI) (generated)
>
> Dataset 2 (industry)
>
> can be used for us? 

## TwinDroid

> [Dataset](https://github.com/AsmaLif/TwinDroid-dataset)
>
> can be used for us? - YES✅

This dataset contains 400 system call traces on Android, which is collected from 151 applications (59 benign and 92 malicious). Each trace contains so many calls (2k+).

*Should we truncate these traces to more smaller ones to get more call traces? we can overlap some tokens at both ends to maintain context continuity.*

This dataset contains syscalls with args in separated traces

```
# dataset snippet
1607111369.438553 getpid()              = 7215
1607111369.438669 gettid()              = 7215
1607111369.438722 clock_gettime(CLOCK_MONOTONIC, {1201552, 61261750}) = 0
1607111369.438891 ioctl(9, 0xc0186201, 0xbffe9338) = 0
1607111369.439661 ioctl(9, 0xc0186201, 0xbffe9338) = 0
1607111369.447369 clock_gettime(CLOCK_MONOTONIC, {1201552, 69804846}) = 0
1607111369.447466 ioctl(9, 0xc0186201, 0xbffe9168) = 0
```

## SNIA-IOTTA

> [Dataset](https://iotta.snia.org/traces/system-call)
>
> can be used for us? - No❌ too old & no diversity

The repository aims for storage-related I/O trace and there is a warning these traces are too old.

The latest dataset (2014) only contains 4 syscalls (open close read write). 

```
# FIU dataset snippet
110257756 1411358400.902384 WRITE     33378457     303104     266240  4096
110257757 1411358400.902561 WRITE     33378458    4202496    3956736  4096
110257758 1411358400.902699 CLOSE     45947061          0
110257759 1411358400.903032 WRITE     33378457     303104     253952  4096
110257760 1411358400.903182 WRITE     33378460   54534144   53637120 12288
110257761 1411358400.903299 CLOSE     45947071          0
110257762 1411358400.903581 CLOSE     45947072          0
```

## DongTing

> [Dataset](https://github.com/HNUSystemsLab/DongTing?tab=readme-ov-file)
>
> can be used for us? - Yes✅

The dataset covers Linux kernels released in the last five years and includes a total of 18,966 normal and attack sequences collected from actual running programs.

```
#dataset snippet
execve|brk|arch_prctl|access|openat|newfstatat|mmap|close|openat|read|pread64|pread64|pread64|newfstatat|mmap|pread64|mmap|mmap|mmap|mmap|mmap|close|mmap|arch_prctl|mprotect|mprotect|mprotect|munmap|mmap|clone|wait4|prctl|clock_nanosleep|setpgid|openat|ioctl|close|close|close|close|close|close|close|close|close|close|close|wait4|close|clock_nanosleep|close|close|close|close|close|close|close|close|close|close|close|close|wait4|close|clock_nanosleep|close|close|exit_group|wait4|clone|wait4|prctl|clock_nanosleep|setpgid|openat|ioctl|close|close|close|close|close|close|close|close|close|close|close|wait4|
```

## SysCall Dataset: A Dataset for Context Modeling and Anomaly Detection using System Calls

> [Dataset](https://data.mendeley.com/datasets/vfvw7g8s8h/2)
>
> can be used for us? - No❌ no diversity

This dataset contains one trace of syscalls with args presented with register values that collected from an uncrewed aerial vehicle (UAV) running on a simulated platform. It has 430k syscalls but 97% of them are the same repeated syscalls.

```
#processed dataset snippest
Beta(difference of adjasent timestamps)	SysCall ID
...
3 24
1	1
2	24
1	1
2	24
...

#raw dataset snippest
Timestamp "SYSCALL" RAX(call ID) RDI RSI RDX R10 R8 R9 CR3
Timestamp "SYSRET" CR3 
592834, SYSCALL,0x17,0x6,0x7ffc7f0817f0,0x7ffc7f081870,0x7ffc7f0818f0,0x0,0x0,0x1f86f000
592834, SYSRET,0x1f86f000
592835, SYSCALL,0x0,0x3,0x7f33335ceeb4,0x1,0x0,0x0,0x0,0x1f86f000
592835, SYSRET,0x1f86f000
592837, SYSCALL,0x17,0x6,0x7ffc7f0817f0,0x7ffc7f081870,0x7ffc7f0818f0,0x0,0x0,0x1f86f000
592837, SYSRET,0x1f86f000
592838, SYSCALL,0x0,0x3,0x7f33335ceeb4,0x1,0x0,0x0,0x0,0x1f86f000
```

## ADFA IDS from UNSW

> [Dataset](https://research.unsw.edu.au/projects/adfa-ids-datasets)
