#!/usr/bin/python3
import tensorflow as tf
import sys
import os

model = sys.argv[1]
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(tf.gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
os.system('rm -f log/*')
summaryWriter = tf.summary.FileWriter('log/', graph)
os.system('tensorboard --logdir log/')
