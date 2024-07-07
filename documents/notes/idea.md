## preprocessor

Run in syz-mamanger.

DFS to check if the arg of a call contains the ResultArg. And preprocessor won't replace this kind of args since can't find the ResultArg addr when parsing it from the AST format to textual.



## SyzLLM-client

The first thing is to replace args from the arg table build in preprocessor. Called NormalizeArgs. So that the given calls can find a corresponding token in server. 

