1. Sequence: Only a call may not invoke a crash, it can trigger that only in certain kernel state, which created by prvious calls. 
   - We can not find a effective way to find the pattern that effectively generate the 'functional crash sequence'. So Healer introduce 'learning relation' to improve coverage so as to approach 'certain kernel state'.

2. It's dangerous to call functions or access data structs without validate. There ofen happen crash. 





1. 如何保证大模型的test cases quality.
2. 使用大模型去逼近healer的idea，并设计一个能够让大模型学习的机制。
3. trade-off between performance and quality.

