#!/usr/bin/env python

import sys
import os
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from inner.data_generator import ImageDataGenerator
from inner.maml import MASF

FLAGS = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## Dataset PACS
flags.DEFINE_integer('num_classes', 2800, 'number of classes used in classification.')
flags.DEFINE_string('outset', 'movie', 'out set genre')

## Training options
flags.DEFINE_string('root', './data/train/pairs', 'log date')
flags.DEFINE_integer('train_iterations', 5000, 'number of training iterations.')
flags.DEFINE_integer('meta_batch_size', 128, 'number of images sampled per source domain')
flags.DEFINE_float('inner_lr', 0.001, 'step size alpha for inner gradient update on meta-train')
flags.DEFINE_float('outer_lr', 0.001, 'learning rate for outer updates with (task-loss + meta-loss)')
flags.DEFINE_float('metric_lr', 0.001, 'learning rate for the metric embedding nn with AdamOptimizer')
flags.DEFINE_float('margin', 10, 'distance margin in metric loss')
flags.DEFINE_bool('clipNorm', True, 'if True, gradients clip by Norm, otherwise, gradients clip by value')
flags.DEFINE_float('gradients_clip_value', 2.0, 'clip_by_value for SGD computing new theta at meta loss')

## Logging, saving, and testing options
flags.DEFINE_string('date', '072720', 'log date')
flags.DEFINE_string('exp', 'mc-maml', 'experiment name')
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', './log', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_integer('summary_interval', 20, 'frequency for logging training summaries')
flags.DEFINE_integer('save_interval', 200, 'intervals to save model')
flags.DEFINE_integer('print_interval', 50, 'intervals to print out training info')
flags.DEFINE_integer('infer_interval', 200, 'intervals to test the model')


def suffule_line(l1, l2, l3):
    """Shuffle 3 list with same shuffle order."""
    lines, l1_new, l2_new, l3_new = [], [], [], []
    for i in range(len(l1)):
        lines.append((l1[i], l2[i], l3[i]))
    random.shuffle(lines)
    for i in range(len(lines)):
        l1_new.append(lines[i][0])
        l2_new.append(lines[i][1])
        l3_new.append(lines[i][2])
    return np.array(l1_new), np.array(l2_new), np.array(l3_new)


def train(model, saver, sess, str_dict, pairs_dir, resume_itr=0):

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + str_dict['log'], sess.graph)
    support_losses, query_losses, support_accuracies, query_accuracies = [], [], [], []

    # Data loaders
    with tf.device('/cpu:0'):
        tr_data_list, train_iterator_list, train_next_list = [],[],[]
        for i in range(len(pairs_dir)):
            # get support set and query set
            pair_name = os.path.basename(pairs_dir[i])
            s_set, q_set = pair_name.split('-')
            s_set_list = os.path.join(pairs_dir[i], s_set+'.txt')
            q_set_list = os.path.join(pairs_dir[i], q_set+'.txt')
            
            s_set_data = ImageDataGenerator(s_set_list, batch_size=FLAGS.meta_batch_size, \
                                            num_classes=FLAGS.num_classes, shuffle=False) 
            q_set_data = ImageDataGenerator(q_set_list, batch_size=FLAGS.meta_batch_size, \
                                            num_classes=FLAGS.num_classes, shuffle=False) 
            s_iterator = tf.data.Iterator.from_structure(s_set_data.data.output_types, \
                                                             s_set_data.data.output_shapes)
            q_iterator = tf.data.Iterator.from_structure(q_set_data.data.output_types, \
                                                             q_set_data.data.output_shapes)

            # there we arrange Support and Query interator in tuples, then list them.
            # so we got a iterator list with shape [n,2], n refers the number of genres.
            tr_data_list.append((s_set_data, q_set_data))
            train_iterator_list.append((s_iterator, q_iterator))
            train_next_list.append((s_iterator.get_next(), q_iterator.get_next()))

    # Ops for initializing different iterators
    training_init_op = []
    s_batches_per_epoch, s_batch_marker = [], []
    q_batches_per_epoch, q_batch_marker = [], []
    for i in range(len(pairs_dir)):
        s_init = train_iterator_list[i][0].make_initializer(tr_data_list[i][0].data)
        q_init = train_iterator_list[i][1].make_initializer(tr_data_list[i][1].data)
        training_init_op.append((s_init, q_init))

        s_batches_per_epoch.append(int(np.floor(tr_data_list[i][0].data_size/FLAGS.meta_batch_size)))
        q_batches_per_epoch.append(int(np.floor(tr_data_list[i][1].data_size/FLAGS.meta_batch_size)))
        s_batch_marker = s_batches_per_epoch[:]
        q_batch_marker = q_batches_per_epoch[:]

    # Training begins
    print("Start training.")
    best_test_acc = 0
    start_time = time.time()
    # Initialize training iterator when itr=0 or it using out
    for i in range(len(pairs_dir)):
       sess.run(training_init_op[i][0])
       sess.run(training_init_op[i][1])
         
    for itr in range(resume_itr, FLAGS.train_iterations):

        # Sample a pair from iterator list
        sampled_index = random.randint(0, len(pairs_dir)-1)
        sampled_pair = train_next_list[sampled_index]
        # print ("Sampled pair: %s \n" % os.path.basename(pairs_dir[sampled_index])), 

        s_batch_marker[sampled_index]  = s_batch_marker[sampled_index]-1
        q_batch_marker[sampled_index]  = q_batch_marker[sampled_index]-1
        if s_batch_marker[sampled_index] <= 0:
            sess.run(training_init_op[sampled_index][0])
            s_batch_marker[sampled_index] = s_batches_per_epoch[sampled_index]
        if q_batch_marker[sampled_index] <= 0:
            sess.run(training_init_op[sampled_index][1])
            q_batch_marker[sampled_index] = q_batches_per_epoch[sampled_index]
        
        # Get sampled data
        inputa0, labela0 ,namea0 = sess.run(sampled_pair[0])
        inputb0, labelb0 ,nameb0 = sess.run(sampled_pair[1])

        s_batch_marker[sampled_index]  = s_batch_marker[sampled_index]-1
        q_batch_marker[sampled_index]  = q_batch_marker[sampled_index]-1
        if s_batch_marker[sampled_index] <= 0:
            sess.run(training_init_op[sampled_index][0])
            s_batch_marker[sampled_index] = s_batches_per_epoch[sampled_index]
        if q_batch_marker[sampled_index] <= 0:
            sess.run(training_init_op[sampled_index][1])
            q_batch_marker[sampled_index] = q_batches_per_epoch[sampled_index]
 
        inputa1, labela1 ,namea1 = sess.run(sampled_pair[0])
        inputb1, labelb1 ,nameb1 = sess.run(sampled_pair[1])

        # shuffle query set
        inputb0, labelb0 ,nameb0 = suffule_line(inputb0, labelb0 ,nameb0)
        inputb1, labelb1 ,nameb1 = suffule_line(inputb1, labelb1 ,nameb1)

        feed_dict = {model.inputa0: inputa0, model.labela0: labela0, \
                     model.inputb0: inputb0, model.labelb0: labelb0, \
                     model.inputa1: inputa1, model.labela1: labela1, \
                     model.inputb1: inputb1, model.labelb1: labelb1, \
                     model.KEEP_PROB: 0.7}

        # self.lossa_raw, self.lossb_raw, accuracya0, accuracyb0, accuracya1, accuracyb1 = result
        output_tensors = [model.task_train_op]
        output_tensors.extend([model.summ_op, model.lossa, model.lossb, model.accuracya0, model.accuracyb0, model.accuracya1, model.accuracyb1])
        _, summ_writer, support_loss, query_loss, a_meta_train_acc, b_meta_train_acc, a_meta_test_acc, b_meta_test_acc = sess.run(output_tensors, feed_dict)

        support_losses.append(support_loss)
        query_losses.append(query_loss)
        support_accuracies.append((a_meta_train_acc+b_meta_train_acc)/2)
        query_accuracies.append((a_meta_test_acc+b_meta_test_acc)/2)

        if itr % FLAGS.print_interval == 0:
            end_time = time.time()
            print('---'*10 + '\n%s' % str_dict['exp'])
            print('time %.4f s' % (end_time-start_time))
            print('Iteration %d' % itr + ': S Loss ' + str(np.mean(support_losses)))
            print('Iteration %d' % itr + ': Q Loss ' + str(np.mean(query_losses)))
            print('Iteration %d' % itr + ': S Accuracy ' + str(np.mean(support_accuracies)))
            print('Iteration %d' % itr + ': Q Accuracy ' + str(np.mean(query_accuracies)))
            support_losses, query_losses = [], []
            support_accuracies, query_accuracies = [], []
            start_time = time.time()

        # save model & make inference.
        if itr == 0:
            saver.save(sess, '{}/{}/model{}'.format(FLAGS.logdir, str_dict['log'], str(itr)))
            os.system('./infer.sh {}  {} 2>&1 | tee -a infer-{}.log'.format(str_dict['log'], FLAGS.outset, str_dict['exp']))
        if (itr!=0) and itr % FLAGS.save_interval == 0:
            saver.save(sess, '{}/{}/model{}'.format(FLAGS.logdir, str_dict['log'], str(itr)))
        if (itr!=0) and itr % FLAGS.infer_interval == 0:
            assert FLAGS.infer_interval % FLAGS.save_interval == 0
            os.system('./infer.sh {} {} 2>&1 | tee -a infer-{}.log'.format(str_dict['log'], FLAGS.outset,str_dict['exp']))


def remove_genre(pairs, outset):
    '''remove outset genre from training data'''
    if outset != 'sm':
        for pair in pairs:
            if outset in pair:
                pairs.remove(pair)
    else:
        for pair in pairs:
            if 'singing' in pair:
                pairs.remove(pair)
        for pair in pairs:
            if 'movie' in pair:
                pairs.remove(pair)

    return pairs        


def main():

    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    # path to .txt files (e.g., art_painting.txt, cartoon.txt) where images are listed line by line
    filelist_root = FLAGS.root
    pairs = os.listdir(filelist_root)

    # remove outset genre from training data
    pairs = remove_genre(pairs, FLAGS.outset)

    str_dict = {}
    str_dict['date'] = FLAGS.date
    str_dict['exp'] = FLAGS.exp
    str_dict['log'] = str_dict['date']+'/'+str_dict['exp']

    # Constructing model
    model = MASF(FLAGS.num_classes, FLAGS.meta_batch_size)
    model.construct_model_train()
    
    model.summ_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=51, var_list=tf.global_variables())
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    
    # resume setting
    resume_itr = 0
    model_file = None
    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + str_dict['log'])
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
    # make list
    pairs_dir = [os.path.join(filelist_root, pair) for pair in pairs]

    # train model
    train(model, saver, sess, str_dict, pairs_dir, resume_itr)


if __name__ == "__main__":
    main()
