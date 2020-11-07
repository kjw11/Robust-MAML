#!/usr/bin/env python
# coding=utf-8

import numpy as np
import os
import shutil
from tqdm import tqdm
from ark2npz_v2 import ark2npz
from tqdm import tqdm


def main():
	ark_file = '/work5/cslt/kangjiawen/070720-maml-cn2/data/raw_data/train_2800/xvector.txt'
	npz_file = '/work5/cslt/kangjiawen/070720-maml-cn2/data/train/xvector.npz'
	# ark2npz
	#ark2npz(ark_file, npz_file)

	npy_dir = '/work5/cslt/kangjiawen/070720-maml-cn2/data/train/npys'
	list_dir = '/work5/cslt/kangjiawen/070720-maml-cn2/data/train/txts'
	# read a dict
	genre_dict = split_genre(npz_file)
	# generate npy file and scp list
	split_npy(genre_dict, npy_dir, list_dir)


def txt_save(filename, data):
	f = open(filename, 'a')
	for i in range(len(data)):
		s = str(data[i]).replace('[','').replace(']','')
		s = s.replace("'",'').replace(',','')+'\n'
		f.write(s)
	f.close()


def split_genre(npz_file):
	'''
	Return a dict: [genre_n] = [(mats_1,label_1), 
                                    (mats_2,label_2), 
                                    ..., 
                                    (mats_m,label_m)]
        Egs. dict = split_genre(xvector.npz)
	'''
	data = np.load(npz_file)

	genre = data['genre']
	mats = data['mats']
	lbs = data['spkers']
	all_genre = np.unique(genre)
	genre_dict = {}

	for i,tar_g in enumerate(all_genre):
		genre_dict[tar_g] = []
		for j,g in enumerate(genre):
			if tar_g == g:
				genre_dict[tar_g].append((mats[j], lbs[j]))
		print tar_g, len(genre_dict[tar_g])
	return genre_dict


def split_npy(genre_dict, npy_dir, list_dir):
	'''
	Split each utterance to xxx.npy file, and return a scp list.
	npy_dir: 
	         - genre_1
	         - genre_2
	            - 2-0.npy
	            - 2-1.npy
	            ...
	list_dir:
	         genre_1.txt
	         genre_2.txt
	         ...
	Egs. split_npy(dict, ./npy/, ./txts/)
	'''
	# clear
	if os.path.exists(npy_dir):
		shutil.rmtree(npy_dir)
	if os.path.exists(list_dir):
		shutil.rmtree(list_dir)
	os.makedirs(list_dir)

	for k, v in genre_dict.items():
		genre = k
		genre_dir = os.path.join(npy_dir, genre)
		os.makedirs(genre_dir)

		txtlst = []
		for idx,(mat, label) in enumerate(tqdm(v)):
			savepath = os.path.join(genre_dir, str(genre)+'-'+str(idx))
			np.save(savepath, mat)
			txtline = savepath+'.npy '+str(label)
			txtlst.append(txtline)
                txt_f_name = os.path.join(list_dir,'{}.txt'.format(genre))
		print "Genre: {}\nDir: {}\nnum_utt: {}\n".format(genre, txt_f_name, len(txtlst))
		txt_save(txt_f_name, txtlst) 

		
if __name__ == "__main__":
	main()

