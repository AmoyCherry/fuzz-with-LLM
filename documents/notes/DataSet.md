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
> can be used for us? - YES✅

This dataset is generated by an Apache server-client app and collected by LTTng in CTF (Common Trace Format). CTF includes a metadata section that describes the binary data in the trace files and uses a subset of the TSDL (Trace Stream Description Language).

```
# metadata snippet
event {
	name = "syscall_entry_write";
	id = 5;
	stream_id = 0;
	fields := struct {
		integer { size = 32; align = 8; signed = 0; encoding= none; base = 10; } _fd;
		integer { size = 64; align = 8; signed = 0; encoding = none; base = 10; } _buf;
		integer { size = 64; align = 8; signed = 0; encoding = none; base = 10; } _count;
	};
};

event {
	name = "syscall_exit_writev";
	id = 312;
	stream_id = 0;
	fields := struct {
		integer { size = 64; align = 8; signed = 1; encoding = none; base = 10; } _ret;
		integer { size = 64; align = 8; signed = 0; encoding = none; base = 10; } _vec;
	};
};
```

This dataset contains 66 million lines of traces in total (both entry and exit and other).

```
# parsed dataset snippet

# syscall_entry marks when to call write with parameters
[04:45:26.499091326] (+0.000004479) server syscall_entry_write: { cpu_id = 0 }, { procname = "bmon", pid = 1771, tid = 1771 }, { fd = 1, buf = 942015091 76096, count = 7 }

# syscall_exit marks when to finish calling write with return value
[04:45:26.499096222] (+0.000004896) server syscall_exit_write: { cpu_id = 0 }, { procname = "bmon", pid = 1771, tid = 1771 }, { ret = 7 }

write @Param@{ fd = 1, buf = 94201509176096, count = 7 }@Param@ @Ret@{ ret = 7 }@Ret@ @Time@04:45:26.499091326@Time@ @Proc@bmon@Proc@
```

### Conversion

#### Phase 1 - merge syscall_entry\_* (parameters) and syscall_exit_\* (return value) into one trace

Goals:

Normalize syscall and args and return value and start time.

Rules: 

1. Extract 

   1. syscall name 
   2. and its parameters 
   3. and its return value 
   4. and entry time 
   5. and ~~process name~~ pid for resource locating;

2. We will encounter entry trace first to get the syscall name and its parameters and entry time;

3. We should find its corresponding return value (syscall_exit_) in the next 1S traces;

4. Target converted data should look like as follow.

   ```
   openat @Param@{ dfd = -100, filename = "/usr/lib/firefox/libXt.so.6", flags = 524288, mode = 0 }@Param@ @Ret@{ ret = 57 }@Ret@ @Time@04:48:53.574837309@time@ @PID@1870@PID@
   
   read @Param@{ fd = 23, count = 1 }@Param@ @Ret@{ ret = 1, buf = 140723345741599 }@Ret@ @Time@04:48:53.579737309@Time@ @PID@1870@PID@
   
   close @Param@{ fd = 12 }@Param@ @Ret{ ret = 0 }@Ret@ @Time@04:48:53.974837309@Time@ @PID@1870@PID@
   
   sendmsg @Param@{ fd = 55, msg = 139820868476640, flags = 16384 }@Param@ @Ret@{ ret = 220 }@Ret@ @Time@04:49:00.574837309@Time@ @PID@1870@PID@
   ```

**Issue**

The trace contains some long consecutive same calls.

```
Consecutive calls greater than 20:
mprotect: 1568 times
read: 285 times
futex: 23980 times
madvise: 2714 times
newlstat: 137 times
rt: 483 times
newstat: 1645 times
close: 133 times
recvmsg: 84 times
poll: 54 times
munmap: 100 times
openat: 26 times
newfstatat: 2 times

Consecutive calls greater than 50:
read: 85 times
futex: 9041 times
madvise: 1255 times
mprotect: 182 times
rt: 262 times
newstat: 736 times
poll: 20 times


Consecutive calls greater than 500:
futex: 2720 times
madvise: 97 times
newstat: 6 times
poll: 3 times
mprotect: 4 times
```

#### Phase 2 - syzkaller format

Goals:

Split traces into programs by time and set resources.

Rules:

1. Target data:

   ```
   [SEP]
   openat$SyzLLM(-100, "/usr/lib/firefox/libXt.so.6", 524288, 0)
   read$SyzLLM(openat$SyzLLM(-100, "/usr/lib/firefox/libXt.so.6", 524288, 0), 1)
   close$SyzLLM(openat$SyzLLM(-100, "/usr/lib/firefox/libXt.so.6", 524288, 0))
   sendmsg$SyzLLM(socket$SyzLLM(1, 524289, 0), 139820868476640, 16384)
   [SEP]
   ```

2. Within every second there are average `3.3e7/180s=183,333` calls. I think the program size is a key factor to the model's performance. And if we choose `30` as the size so we need to split programs every `1s/(183,333/30)=0.00016s=0.16ms`.

   - We should focus on the later stage of fuzzing where run larger programs than the initial stage like 10 to 20 calls per program. But I found the program sizes have not much change compare with the initial stage and rare programs have more than 20 calls.

3. We should search for resources by fd and return value and use resources to replace consumers's fd.

   - some fd may mismatch.

#### Phase 3 - training format

Goals:

Normalize args for training.

Rules:

1. There are only three formats of args in the training data - resource, addr, constant. 

   - The constants are meaningful (e.g. flag, mode) and we should retain and convert them to hex. 
   - Addresses are related to strings (e.g. path) and data structures (e.g. msg) and we can assign to them.
   - Fallback: Assign resources if not any match.

2. Target data:

   ```
   [SEP]
   openat$SyzLLM(0xffffffffffffff9c, &(0x7f0000008000)='./file0\x00', 0x80000, 0x0)
   read$SyzLLM(@RSTART@openat$SyzLLM(0x0, &(0x7f0000008000)='/proc/keys\x00', 0x1, 0x0)@REND@, &(0x7f0000017000)=""/64, 0x1)
   close$SyzLLM(@RSTART@openat$SyzLLM(0x0, &(0x7f0000008000)='/proc/keys\x00', 0x1, 0x0)@REND@)
   sendmsg$SyzLLM(@RSTART@socket$SyzLLM(0x1, 0x1, 0x1)@REND@, (0x7f000000f000)={&(0x7f000000f400)=nil, 0x1, &(0x7f000000f800)={&(0x7f000000fc00)=@newlink={0x1, 0x1, 0x401, 0x1, 0x1, {0x1, 0x1, 0x1}, [@IFLA_MASTER={0x8, 0x3}, @IFLA_LINKINFO={0x20, 0x12, 0x0, 0x1, @bond={{0x9}, {0x10, 0x2, 0x0, 0x1, [@IFLA_BOND_ARP_IP_TARGET={0x4}, @IFLA_BOND_ARP_INTERVAL={0x8, 0x7, 0x6}]}}}]}, 0x1}, 0x1, 0x1, 0x1, 0x0}, 0x4000)
   [SEP]
   ```

#### Phase4 - Collect and assign addresses and struct values

Goals:

Collect addresses associated with their struct values from syzkaller's dataset and assign them to the new data set.

Rules:

1. Collect addresses from vocab.txt;
   - Filter addresses that with res;
   - Map: {call_name}$SyzLLM_{arg_len}[{addr_index}] = addr;
2. Assign pointer values;
   - Pointers have the tags in phase1.txt: `uaddr`, `addr`, `buf`, `buff`, `ubuf`, `arg`(not all)... 
   - Consider all numbers that greater than `140000000000000` exclude `-1` are addresses.
   - We should not assign pointer values to all where is address type, some addr are 0 also is meaningful. We only care about the addr above `140000000000000`.

## TwinDroid

> [Dataset](https://github.com/AsmaLif/TwinDroid-dataset)
>
> can be used for us? - YES✅

This dataset contains 400 system call traces on Android, which is collected from 151 applications (59 benign and 92 malicious). Each trace contains so many calls (2k+).

*Should we truncate these traces to more smaller ones to get more call traces? we can overlap some tokens at both ends to maintain context continuity.* Group traces within a reasonable interval like 500ms into programs.

This dataset contains syscalls with args in separated traces.

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
> can be used for us? - NO❌ too old & no diversity

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
> can be used for us? - YES✅ low priority

The dataset covers Linux kernels released in the last five years and includes a total of 18,966 normal and attack sequences collected from actual running programs.

```
#dataset snippet
execve|brk|arch_prctl|access|openat|newfstatat|mmap|close|openat|read|pread64|pread64|pread64|newfstatat|mmap|pread64|mmap|mmap|mmap|mmap|mmap|close|mmap|arch_prctl|mprotect|mprotect|mprotect|munmap|mmap|clone|wait4|prctl|clock_nanosleep|setpgid|openat|ioctl|close|close|close|close|close|close|close|close|close|close|close|wait4|close|clock_nanosleep|close|close|close|close|close|close|close|close|close|close|close|close|wait4|close|clock_nanosleep|close|close|exit_group|wait4|clone|wait4|prctl|clock_nanosleep|setpgid|openat|ioctl|close|close|close|close|close|close|close|close|close|close|close|wait4|
```

## SysCall Dataset: A Dataset for Context Modeling and Anomaly Detection using System Calls

> [Dataset](https://data.mendeley.com/datasets/vfvw7g8s8h/2)
>
> can be used for us? - NO❌ no diversity

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
>
> can be used for us?
