This dir contains codes for **multi-condition training** experiments.

Introduction:
=============

 - main.py is the main file, which reads a all.txt as input. Note that the
   out-set genre should be removed in advance. Refernce data:
       - /work5/cslt/kangjiawen/091620-maml-cn2/data/train/mct_data/genre
       - or /work103/kangjiawen/091620-maml-cn2/data/train/mct_data/genre
   for partial data:
       - /work5/cslt/kangjiawen/091620-maml-cn2/data/train_mc/mct_data/genre
       - or /work103/kangjiawen/091620-maml-cn2/data/train_mc/mct_data/genre

 - infer.sh is a recipe to inferance vectors from model, and calculate EERs
   for different genres. Note that the enroll and test is matched condition.

Implement:
==========
If you want to make a quick running, you can:

    ln -s /work5/cslt/kangjiawen/070720-maml-cn2/data

to make align (the original file is rather big), and:

    ./test.sh    

note that you can adjust the parameters in test.sh 
to speed up.


Source:
=======
This code is revised from: https://github.com/biomedia-mira/masf
