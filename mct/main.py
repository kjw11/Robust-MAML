#!/usr/bin/env python

import sys
import os
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from inner.data_generator import ImageDataGenerator
from inner.maml_mct import MASF

FLAGS = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## Dataset PACS
flags.DEFINE_integer('num_classes', 2800, 'number of classes used in classification.')
flags.DEFINE_string('outset', 'movie', 'log date')

## Training options
flags.DEFINE_string('root', './data/train/mct_data/genre', 'log date')
flags.DEFINE_integer('train_iterations', 2000, 'number of training iterations.')
flags.DEFINE_integer('meta_batch_size', 128, 'number of images sampled per source domain')
flags.DEFINE_float('inner_lr', 0.001, 'step size alpha for inner gradient update on meta-train')
flags.DEFINE_float('outer_lr', 0.001, 'learning rate for outer updates with (task-loss + meta-loss)')
flags.DEFINE_float('metric_lr', 0.001, 'learning rate for the metric embedding nn with AdamOptimizer')
flags.DEFINE_float('margin', 10, 'distance margin in metric loss')
flags.DEFINE_bool('clipNorm', True, 'if True, gradients clip by Norm, otherwise, gradients clip by value')
flags.DEFINE_float('gradients_clip_value', 2.0, 'clip_by_value for SGD computing new theta at meta loss')

## Logging, saving, and testing options
flags.DEFINE_string('date', '072720', 'log date')
flags.DEFINE_string('exp', 'mc-mct', 'experiment name')
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


def train(model, saver, sess, str_dict, train_list, resume_itr=0):

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + str_dict['log'], sess.graph)
    losses, accs = [], []

    # Data loaders
    with tf.device('/cpu:0'):
        inp_data = ImageDataGenerator(train_list, batch_size=FLAGS.meta_batch_size, \
                                            num_classes=FLAGS.num_classes) 
        iterator = tf.data.Iterator.from_structure(inp_data.data.output_types, \
                                            inp_data.data.output_shapes)
        next_batch = iterator.get_next()
    init_op = iterator.make_initializer(inp_data.data)
    batches_per_epoch = int(np.floor(inp_data.data_size/FLAGS.meta_batch_size))

    # Training begins
    print("Start training.")
    start_time = time.time()     
    for itr in range(resume_itr, FLAGS.train_iterations):

        # Initialize training iterator when itr=0 or it using out
        if FLAGS.resume:
            sess.run(init_op)
            FLAGS.resume = False
        if itr == 0:
            sess.run(init_op)
        if itr % batches_per_epoch == 0:
            print('reset data loader')
            sess.run(init_op)

        # Get sampled data
        input, label, names = sess.run(next_batch)

        feed_dict = {model.input: input, model.label: label, \
                     model.KEEP_PROB: 0.7}

        output_tensors = [model.task_train_op]
        output_tensors.extend([model.loss, model.accuracy])
        _, loss, acc = sess.run(output_tensors, feed_dict)

        losses.append(loss)
        accs.append(acc)

        if itr % FLAGS.print_interval == 0:
            end_time = time.time()
            print('---'*10 + '\n%s' % str_dict['exp'])
            print('time %.4f s' % (end_time-start_time))
            print('Iteration %d' % itr + ': Loss ' + str(np.mean(losses)))
            print('Iteration %d' % itr + ': Accuracy ' + str(np.mean(acc)))
            losses = []
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

        
def main():

    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    # path to .txt files (e.g., speech.txt, singing.txt) where data are listed line by line
    train_dir = FLAGS.root
    train_list = os.path.join(FLAGS.root ,"all_{}.txt".format(FLAGS.outset)) # "all" means all genre together

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

    # train model
    train(model, saver, sess, str_dict, train_list, resume_itr)


if __name__ == "__main__":
    main()
