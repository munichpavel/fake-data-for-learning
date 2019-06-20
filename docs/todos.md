# Fake data for learning TODOs

Test independence, binary model
Test binary conditional embeddings of 2 vars, X0 -> X1

* ~~write test for X0~~
* ~~dummy implementation of GPT for X0~~
* ~~real first but flawed implementation for X0~~
* ~~write test for X1 | X0~~
* ~~write test for list ranking~~
* ~~write test for reindexing~~
* ~~refactor implementation for embeddings, including~~
* ~~eliminating list ranking and reindexing methods~~

~~Test embeddings binary X0 -> X1 -> X2~~

Write sanity checks for graph probability tables, e.g.
* I, J, K partition[m] for some m
~~Package arguments better: s and [m] are same for all GPTs in BN (see refactor TODO below)~~

## Branch local-tables
~~remove embedding tests, as this is non-local~~
~~refactor interface so only local input needed~~
~~add fake extra arguments, e.g. node_var, input_vars (DROPPED)~~
~~test set xk, rv outcomes~~
~~test pmf outcomes~~
~~at least somewhat meaningful test that can call random~~
extend class to conditional rvs
* make init work
* ~~add somewhat meaningful rvs test~~
* make rvs work
* ~~keep real pmf if no parents, else overwrite with dummy pmf (DEPPED)~~
* ~~dummy make pmf as a function of input vars (DEPPED)~~
* ~~change arg names (DEPPED)~~
* ~~make pmf real (DEPPED)~~

* add test for valueerror if dims of pt and var + parent vars incompatible
