#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: ark2npz_v2.py
#	> Author: Yang Zhang, Jiawen Kang 
#	> Comment: This file is revised from ark2npz.py, only used for CN-Celeb
#		dataset. It generate npz file with spker and genre label and 
#		without	the requirement of utt2spk file.
# ************************************************************************/


import numpy as np
import kaldi_io


def ark2npz(ark_path, npz_path):
	'''load ark data format and save as npz data format

	label: spker.shape=(utt_num, )
	data : feats.shape=(utt_num, 72)

	//load the data and label
	feats = np.load(npz_path, allow_pickle=True)['feats']
	spkers = np.load(npz_path, allow_pickle=True)['spkers']

	'''
	print("ark data loading...")
	utts = []
	mats = []
	for k, v in kaldi_io.read_mat_ark(ark_path):
		utts.append(k)
		mats.append(v)
	counter = 0
	feats = []
	genre = []
	spkers = []
	for mat in mats:
		for i in mat:
			feats.append(i)
			genre.append(utts[counter].split('-')[1].encode('gbk'))
			spkers.append(utts[counter][:7])
		counter+=1
	
	# convert string-label to num label
	string_lable = spkers
	spkers = np.unique(spkers)

	index = 0
	table = {}
	for it in spkers:
		table[it] = index
		index+=1
	
	num_label=[]
	for spk in string_lable:
		num_label.append(table[spk])
		
	print("saving...")
	np.savez(npz_path, mats=feats, spkers=num_label, genre=genre)
	print("sucessfully convert {} to {} ".format(ark_path, npz_path))
	print("ark->npz down")


if __name__=="__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--src_file', default="feats.txt", help='src file of feats.(txt)')
	parser.add_argument('--dest_file',default="feats.npz", help='dest file of feats.(npz)')
	args = parser.parse_args()

	ark2npz(args.src_file, args.dest_file)

	# test
	print("\n\ntest...\n")
	mats = np.load(args.dest_file, allow_pickle=True)['mats']
	spkers = np.load(args.dest_file, allow_pickle=True)['spkers']
	genre = np.load(args.dest_file, allow_pickle=True)['genre']

	print("mats shape: ", np.shape(mats))
	print("spker label shape: ", np.shape(spkers))
	print("genre shape: ", np.shape(genre))
	print("num of spker: ", np.shape(np.unique(spkers)))
	print("num of genre: ", np.shape(np.unique(genre)))

	print(spkers[0])
	print(mats[0])
	print(genre[0])

# 	spk = np.unique(spkers)
# 	for it in spk:
# 		print(it)
