This dir contains codes for **robust-maml** experiments.

Introduction:
=============

 - main.py is the standard robust-maml, trained with pair data. utils/make_pairs.py
   is helpful for data preparation. Also refre:
       - /work5/cslt/kangjiawen/091620-maml-cn2/data/train/pairs
       - or /work103/kangjiawen/091620-maml-cn2/data/train/pairs

 - main_pertial.py is a robust-maml trained with partial data, which means
   the data from different genre but same spk will be labeled as different spks.
   utils/make_npy_list.py is helpful for data preparation. Alsorefer:
       - /work5/cslt/kangjiawen/070720-maml-cn2/data/train_mc/txts
       - or /work103/kangjiawen/091620-maml-cn2/data/train_mc/txts 

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
