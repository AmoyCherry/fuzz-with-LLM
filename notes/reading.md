## Healer - SOSP 2021

> [HEALER - Relation Learning Guided Kernel Fuzzing](http://www.wingtecher.com/themes/WingTecherResearch/assets/papers/healer-sosp21.pdf)

### Idea

"In order to reduce the search space and improve fuzzing efficiency, we propose to consider the **influence relations** between system calls during test case **generation** and **mutation**."

### Design

![image-20211129200131315](/Users/wangwenzhi/Downloads/image-20211129200131315.png)

#### static learning

analyze return values and params by detecting syzlang files.

- R(i, j) = 0: no relation.
- R(i, j) = 1: Ci -> Cj, which means Ci's output (func return or ptr param which its type equal to return value's) is Cj's input.

#### dynamic learning

- algo1 - Sequence minimazition: for a syscall sequence, try to delete every syscall and check if the coverage is smaller, if so, the deleted syscall is critical so need to be retained. And the syscall should be deleted if the coverage has no change. 
  - The only goal is to delete the syscalls which don't impact the coverage.
- algo2 - relation learning: 
  1. traverse all syscalls in the sequence in turn;
  2. assume Ci is the previous call of Cj, if Cj's coverage will be smaller atfer delete Ci, we can be sure that there is a relation between Ci and Cj, so that R(i, j) = 1.



## PATA - S&P 2022

> [PATA: Fuzzing with Path Aware Taint Analysis](http://www.wingtecher.com/themes/WingTecherResearch/assets/papers/sp22.pdf)

### Goals

Enhance the efficiency that identify and take advantage of the influencing input bytes (critical bytes) when encountered loops.

### BG

#### the inﬂuencing input bytes

The specific bytes or portions of the input data that have the potential to <u>affect the behavior of the target program</u>.

- passing through different if-conditions and trigger different code paths;
- uncover vulnerabilities or cause unexpected behavior.

Benefits: helps guide the fuzzer towards exploring different program paths and increasing the chances of discovering vulnerabilities.

**Follow up:**

*How to use it to increase the chance of uncover vnbs?*

- guided the mutation process, i.e. focus on operating the influencing input bytes.



#### variable occurrence

```c++
// define variable
int name = get();

// occurrence 1
if (name ...) { }

// occurrence 2
func(C)
```



#### RVS

An excution path represent by a list of variable occurrences e.g. V1 -> V2 -> V1.

### Motivition and Example



![image-20231018220141736](/Users/wangwenzhi/Library/Application Support/typora-user-images/image-20231018220141736.png)

*When execution paths alter after input perturbation, PATA utilizes a matching algorithm to determine which constraint variable occurrence after perturbation matches with a constraint variable occurrence in the original path. Matched pairs are marked with Xin the ﬁgure.*



### Basic Idea

1. Collect RVS of a excution path. 
   1. Collect occurrences of each variable;   
   2. Gen a original input;
   3. Excute input and records values into RVS along the path. 
2. Perturb input bytes and records a list of new values into RVS.
   - Maybe not every variable in RVS could records a new value since a potential excution path could be explored.

3. Match the variable occurrences between two RVSs to check if a new excution path is explored.
4. Compare the values between the matched occurrences to get a list of whether the altered byte is critical to every occurrence (nodes of a path: V1 -> V2 -> V1).



### Comments

Improve the accuracy of locate critical bytes for each variable occurrence with limited memory cost added compare to conventional methods.



## AFLFast - CCS 2016

### the bad in conventional methods (AFL) 

![IMG_8598](/Users/wangwenzhi/Downloads/IMG_8598.jpeg)

#### AFL algorithm

![image-20231029223405640](/Users/wangwenzhi/Library/Application Support/typora-user-images/image-20231029223405640.png)

#### defects

the two methods in line 7 and 8 could be improved.

1. traditional methods dont consider the number of gen-ed inputs (energy) , so ofen cause too much `energe` is assigned for some crash paths which don't required such more, and insufficient for other crash paths which need more. 
2. traditional methods dont consider the order that choose seeds to mutate and fuzz from the queue.





### Idea

make the inputs gen-ed from the low density region can be excute first.

> $t_i$ : a seed which been choosen and will exercises path $i$.
>
> $s(i)$ : the number of $t_i$ has been selected.
>
> $f(i)$ : the number of gen-ed inputs that exercise path $i$
>
> $u$ : the mean number of $f(i)$
>
> $M$ : upper bound of $P$ for current iteration 
>
> the meaning of path $i$ seems equal to 'seed'.



#### Power Schedules

##### AFL

$a(i)$ : depending on the execution time, block transition coverage, and creation time of $t_i$ without consider the frequency of the choosen seed.

$B$ : a constant to reduce the gen-ed inputs size.



##### Cut-Off Exponential (COE)

![image-20231029230029548](/Users/wangwenzhi/Library/Application Support/typora-user-images/image-20231029230029548.png)



##### The exponential schedule (FAST)

1. inversely proportional to the amount of fuzz $f(i)$ that exercises path $i$ includes $f(i) > u$.
2. if $t_i$ has been picked up many times (with a high $s(i)$) we could be more confident that $i$ live in a low-density region.

![image-20231029230044403](/Users/wangwenzhi/Library/Application Support/typora-user-images/image-20231029230044403.png)





Understanding the *Fast*

AFLFast can only explore bugs that AFL can and just finds same number bugs in a shorter time, since AFLFast doesn't modify the AFL'mutator. This is called *merely impact AFL’s efficiency (i.e., #paths explored per unit time), not its effectiveness (i.e., #paths explored in expectation).*