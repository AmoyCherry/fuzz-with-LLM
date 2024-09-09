### Reduce token size

1. I believe the `$desctiption` part in syscalls is just a log for debugging without any functionality. I would normalize it when preprocessing.

2. Remove step increments for different calls for addresses.
3. change `0x111` to `0x1`.

> notes-Aug-19-2024
>
> 1. top k -> return all k calls
>
> 2. stabilize res inline
> 3. DARPA

## Preprocessing

Each syscall consists of a specific syscall name and several args. And a bit of difference in args can result in different syscalls, even though these args are randomly generated and with limited functional information to themselves. Moreover, too many different tokens hide relations between syscalls in programs which harms model training to find more patterns that improve coverage. 

On the other hand, the **flag** args should be considered as a part of the 'specific syscall name' which means syscalls with different flags (**regardless of order**) should be treated as different. 

So the first problem is to solve which bunch of arg types are not important and can be normalize, and others should be kept.

### Should be normalize

1. constArg
   - length;
2. pointerArg
   - All addrs. 





To reduce the token size of model, we map all non-flag (address) args into a pre-allocate addr list. There are four types of non-flag args to process.



Maintain four tables per program.

- `buffer`: [address → flag_index]
  - real arg value as the flag_index;
- `path`: [string → flag_index]
  - real arg value as the flag_index;
- `resource`: [fd → flag_index]
- `res_def`: [flag_index → token]
- `ptr`:[address -> flag_index]
  - `Res`:



Buffer

1. path;
2. meaningless string;

Const:

1. flags, mode,...
2. meaningless num, len,...



Given the scenario:

Call-A (addr-1) -> Call-A (S1)

Call-T (addr-1) -> Call-A (S1)

Call-A (addr-2) -> Call-A (S2)



Call-A (addr-1) -> Call-A (S1)

Call-T (addr-1) -> Call-A (S2)

Call-A (addr-2) -> Call-A (S1)



Maintain global tables as the replacement flags.



nested struct, union and ptr



get a legal call from syzLLM so that it can be deserialized to `prog.Program`. After that, we can use the origin args from syzLLM or gen new args by `prog.Meta`.

### Enums (flags)

Since all the flags of sycalls in linux are integers (like a set), the only thing we need to do is to keep them.

#### Use open as an example

The open call in real program:

> ```c
> /*     linux-5.4/include/uapi/asm-generic/fcntl.h     */
> #define O_RDONLY	00000000
> #define O_WRONLY	00000001
> #define O_RDWR		00000002
> ...
>   
> #define O_CREAT		 01000	/* not fcntl */
> #define O_TRUNC		 02000	/* not fcntl */
> #define O_EXCL		 04000	/* not fcntl */
> #define O_NOCTTY	010000	/* not fcntl */
> ...
> 
> int open(const char *pathname, int flags, mode_t mode);
> ```

```c
const char *filename = "example.txt";  
// Open the file for writing, create if it doesn't exist, and set permissions to 0644
int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
```

The open call in our corpus that to be trained:

```c
r1 = open(&(0x7f0000000200)='./file0\x00', 0x210c2, 0x0)
```

All the flags are integers that obtained by ORing these flags. That stores all the flags without their order which is what we want.

### Identifiers

#### sendmsg

> ```c
> ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags);
> struct msghdr {
>     void          *msg_name;        // protocol address
>     socklen_t      msg_namelen;     // size of protocol address
>     struct iovec  *msg_iov;         // scatter/gather array
>     int            msg_iovlen;      // elements in msg_iov
>     size_t        msg_iovlen;     // ancillary data (cmsghdr struct)
>     socklen_t      msg_controllen;  // length of ancillary data
>     int            msg_flags;       // flags returned by recvmsg()
> };
> ```

```c
sendmsg$key(0xffffffffffffffff, &(0x7f0000000140)={0x0, 0x0, &(0x7f0000000100)={&(0x7f0000000680)=ANY=[@ANYBLOB="02000000bf0164d1a28500000000042774602f36cddb4aa287b3b3312d91f7fbcd26167f6444b666b5023d6da31997c5864183bb5548c8d5210899d6b5b6d5efcd76ffd06e3e62e26c761a6047d1"], 0x4e}}, 0x0)
```

- Identifier - struct msghdr *msg: {0x0, 0x0, &(0x7f0000000100)={&(0x7f0000000680)=ANY=[@ANYBLOB="02000000bf0164d1a28500000000042774602f36cddb4aa287b3b3312d91f7fbcd26167f6444b666b5023d6da31997c5864183bb5548c8d5210899d6b5b6d5efcd76ffd06e3e62e26c761a6047d1"], 0x4e}};

- Address: 0x7f0000000140;

#### timer_settime

> ```c
> int timer_settime(timer_t timerid, int flags,
>                          const struct itimerspec *restrict new_value,
>                          struct itimerspec *_Nullable restrict old_value);
> struct timespec {
>                time_t tv_sec;                /* Seconds */
>                long   tv_nsec;               /* Nanoseconds */
> };
> 
> struct itimerspec {
>                struct timespec it_interval;  /* Interval for periodic timer */
>                struct timespec it_value;     /* Initial expiration */
> };
> ```

```c
timer_settime(0x0, 0x0, &(0x7f0000000000)={{0x0, 0x989680}, {0x0, 0x989680}}, 0x0)
```

- Identifier - struct itimerspec *restrict: {{0x0, 0x989680}, {0x0, 0x989680}};

- Address: 0x7f0000000140;

### Buffers

#### read

> ```c
> ssize_t read(int fd, void buf[.count], size_t count);
> ```

```c
read(r2, &(0x7f0000000080)=""/238, 0xee)
```

#### write

> ```c
> ssize_t write(int fd, const void buf[.count], size_t count);
> ```

```c
write(r3, &(0x7f0000000340), 0x41395527)
```

#### mmap

> ```c
> void *mmap(void addr[.length], size_t length, int prot, int flags, int fd, off_t offset);
> ```

```c
mmap(&(0x7f00009fd000/0x600000)=nil, 0x600000, 0x0, 0x6031, 0xffffffffffffffff, 0x0)
```

### Resources



## SyzLLM should be improved at two levels:

### Effectiveness

1. To train more patterns and catch more information;
   - Training with **all args** cause:
     1. too many different tokens;
     2. too less sentences (corpus) relatively
     3. Not all syscalls from syzkaller have been trained.
2. ncalls -> choice table + SyzLLM
   1. choice table learn relation from worked sequence feed by SyzLLM;
   2. improve performace from pure SyzLLM;
3. SyzLLM return top2 calls randomly; 



### Performance

The main cost is coming from the model inference. It could be improved by **reduce the parameters size**.

Every SyzLLM request cost about 65ms.

1. Communication
   1. Async request;
   2. Using Get instead of Post;



## Model

### Diversity

**Temperature and Top-k sampling**

**Nucleus Sampling (Top-p Sampling)**

