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
* ~~keep real pmf if no parents, else overwrite with dummy pmf (DEPPED)~~
* ~~dummy make pmf as a function of input vars (DEPPED)~~
* ~~change arg names (DEPPED)~~
* ~~make pmf real (DEPPED)~~

MERGED INTO MASTER 

## Forget about derived class from scipy

~~Will copy api of rv_discrete, but subclassing is not worth the trouble (DEPPED)~~
Will not copy rv_discrete, just use np.random.choice

* ~~make init work (DEPPED)~~
* ~~add somewhat meaningful rvs test~~
* ~~dummy implementation of rvs~~
* make rvs work
* ~~allow non-default values~~
* ~~add get_pt method~~
* add test for valueerror if dims of pt and var + parent vars incompatible

Extend to conditional rvs
* ~~Dummy implementation~~
* ~~Non-dummy implementation~~

## Sample from network

* ~~Test for FakeDataBayesianNetwork init, vars~~
  * ~~dummy implementation~~
  * ~~real implementation~~
  * ~~test for vars in parents, missing as rvs, value error~~

* Test for adjacency matrix
  * ~~dummy implementation~~
  * ~~real implementation~~
    * ~~name in list~~
    * ~more tests~
    * refactor

* ~~Refactor cpt definition for easier data entry if more than one parent~~

* Test for sampling using adjacency matrix
  * test for eve node names
    * ~~Add equality method for bayenian network rvs (Not needed in retrospect)~~
    * ~~rename parent argument to parent_names, because that is what it is~~
  * Test sampling eves (maybe to be depped)
    * ~~Dummy implementation~~
    * ~~Real implemnentation
    * ~~refactor (e.g. using adjancy matrix)~~
      * ~~add find zero col idx util function~~
        * ~~dummy implementation (oops, I forgot)~~
        * ~~real implementation~~
  * Test full sampling
