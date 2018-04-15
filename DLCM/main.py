"""Training and testing the Deep Listwise Context Model.

See the following paper for more information.
	
	* Qingyao Ai, Keping Bi, Jiafeng Guo, W. Bruce Croft. 2018. Learning a Deep Listwise Context Model for Ranking Refinement. In Proceedings of SIGIR '18
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange	# pylint: disable=redefined-builtin
import tensorflow as tf
import data_utils

from RankLSTM_model import RankLSTM


#rank list size should be read from data
tf.app.flags.DEFINE_string("data_dir", "/tmp/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./tmp/", "Training directory.")
tf.app.flags.DEFINE_string("test_dir", "./tmp/", "Directory for output test results.")
tf.app.flags.DEFINE_string("hparams", "", "Hyper-parameters for models.")

tf.app.flags.DEFINE_integer("batch_size", 256,
							"Batch size to use during training.")
tf.app.flags.DEFINE_integer("embed_size", 1024,
							"Size of each model layer (hidden layer input size).")
tf.app.flags.DEFINE_integer("max_train_iteration", 0,
							"Limit on the iterations of training (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
							"How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("boost_max_num", 50,
							"The max number of new data for boosting one training instance.")
tf.app.flags.DEFINE_integer("boost_swap_num", 10,
							"How many time to swap when boosting one training instance.")
tf.app.flags.DEFINE_boolean("decode", False,
							"Set to True for decoding data.")
tf.app.flags.DEFINE_boolean("decode_train", False,
							"Set to True for decoding training data.")
tf.app.flags.DEFINE_boolean("feed_previous", False,
                            "Set to True for feed previous internal output for training.")
tf.app.flags.DEFINE_boolean("boost_training_data", False,
                            "Boost training data througn swapping docs with same relevance scores.")



FLAGS = tf.app.flags.FLAGS


def create_model(session, data_set, forward_only):
	"""Create translation model and initialize or load parameters in session."""
	expand_embed_size = max(FLAGS.embed_size - data_set.embed_size, 0)
	model = RankLSTM(data_set.rank_list_size, data_set.embed_size,
					 expand_embed_size, FLAGS.batch_size, FLAGS.hparams,
					 forward_only, FLAGS.feed_previous)

	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
	if ckpt:
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.global_variables_initializer())
	return model


def train():
	# Prepare data.
	print("Reading data in %s" % FLAGS.data_dir)
	
	train_set = data_utils.read_data(FLAGS.data_dir, 'train')
	if FLAGS.boost_training_data:
		print('Boosting training data')
		train_set.boost_training_data(FLAGS.boost_max_num, FLAGS.boost_swap_num)
	valid_set = data_utils.read_data(FLAGS.data_dir, 'valid')
	print("Rank list size %d" % train_set.rank_list_size)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		# Create model.
		print("Creating model...")
		model = create_model(sess, train_set, False)
		print("Created %d layers of %d units." % (model.hparams.num_layers, FLAGS.embed_size))

		# Create tensorboard summarizations.
		train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train_log',
										sess.graph)
		valid_writer = tf.summary.FileWriter(FLAGS.train_dir + '/valid_log')

		#pad data
		train_set.pad(train_set.rank_list_size, model.hparams.reverse_input)
		valid_set.pad(valid_set.rank_list_size, model.hparams.reverse_input)


		# This is the training loop.
		step_time, loss = 0.0, 0.0
		current_step = 0
		previous_losses = []
		best_perplexity = None
		while True:
			# Get a batch and make a step.
			start_time = time.time()
			encoder_inputs, embeddings, decoder_targets, target_weights, target_initial_scores, _ = model.get_batch(train_set.initial_list,
																				train_set.gold_list, train_set.gold_weights,
																				train_set.initial_scores, train_set.features)
			_, step_loss, _, summary = model.step(sess, encoder_inputs, embeddings, decoder_targets,
										target_weights, target_initial_scores, False)
			train_writer.add_summary(summary, current_step)
			step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
			loss += step_loss / FLAGS.steps_per_checkpoint
			current_step += 1


			# Once in a while, we save checkpoint, print statistics, and run evals.
			if current_step % FLAGS.steps_per_checkpoint == 0:

				print(model.start_index)
				train_writer.add_summary(summary, current_step)
				
				# Print statistics for the previous epoch.
				perplexity = math.exp(loss) if loss < 300 else float('inf')
				print ("global step %d learning rate %.4f step-time %.2f perplexity "
							 "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
												 step_time, perplexity))

				# Decrease learning rate if no improvement was seen over last 3 times.
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)
				
				if perplexity == float('inf'):
					break

				# Validate model
				it = 0
				count_batch = 0.0
				valid_loss = 0
				while it < len(valid_set.initial_list) - model.batch_size:
					encoder_inputs, embeddings, decoder_targets, target_weights, target_initial_scores, cache = model.get_next_batch(it, valid_set.initial_list,
																				valid_set.gold_list, valid_set.gold_weights,
																				valid_set.initial_scores, valid_set.features)
					_, v_loss, results, summary = model.step(sess, encoder_inputs, embeddings, decoder_targets,
													target_weights, target_initial_scores, True)
					it += model.batch_size
					valid_loss += v_loss
					count_batch += 1.0
				valid_writer.add_summary(summary, current_step)
				valid_loss /= count_batch
				eval_ppx = math.exp(valid_loss) if valid_loss < 300 else float('inf')
				print("  eval: perplexity %.2f" % (eval_ppx))

				# Save checkpoint and zero timer and loss.
				if best_perplexity == None or best_perplexity >= eval_ppx:
					best_perplexity = eval_ppx
					checkpoint_path = os.path.join(FLAGS.train_dir, "RankLSTM.ckpt")
					model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				
				output_label = results[0][0]
				
				original_input = [encoder_inputs[l][0] for l in xrange(model.rank_list_size)]
				print('ENCODE INPUTS')
				print(original_input)
				print('ENCODE CONTENT')
				print(cache[0])
				print('DECODER WEIGHT')
				print(cache[1])
				print('DECODER TARGET')
				gold_output = [decoder_targets[l][0] for l in xrange(model.rank_list_size)]
				print(gold_output)
				print('OUTPUT')
				print(output_label)

				step_time, loss = 0.0, 0.0
				sys.stdout.flush()

				if FLAGS.max_train_iteration > 0 and current_step > FLAGS.max_train_iteration:
					break



def decode():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		# Load test data.
		print("Reading data in %s" % FLAGS.data_dir)
		test_set = None
		if FLAGS.decode_train:
			test_set = data_utils.read_data(FLAGS.data_dir,'train')
		else:
			test_set = data_utils.read_data(FLAGS.data_dir,'test')

		# Create model and load parameters.
		model = create_model(sess, test_set, True)
		model.batch_size = 1	# We decode one sentence at a time.

		test_set.pad(test_set.rank_list_size, model.hparams.reverse_input)

		rerank_scores = []

		# Decode from test data.
		for i in xrange(len(test_set.initial_list)):
			encoder_inputs, embeddings, decoder_targets, target_weights, target_initial_scores = model.get_data_by_index(test_set.initial_list,
																							test_set.gold_list,
																							test_set.gold_weights,
																							test_set.initial_scores,
																							test_set.features, i)
			_, test_loss, output_logits, summary = model.step(sess, encoder_inputs, embeddings, decoder_targets,
											target_weights, target_initial_scores, True)

			#The output is a list of rerank index for decoder_inputs (which represents the gold rank list)
			rerank_scores.append(output_logits[0][0])
			if i % FLAGS.steps_per_checkpoint == 0:
				print("Decoding %.2f \r" % (float(i)/len(test_set.initial_list))),

		#get rerank indexes with new scores
		rerank_lists = []
		for i in xrange(len(rerank_scores)):
			scores = rerank_scores[i]
			rerank_lists.append(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))

		if FLAGS.decode_train:
			data_utils.output_ranklist(test_set, rerank_lists, FLAGS.test_dir, model.hparams.reverse_input, 'train')
		else:
			data_utils.output_ranklist(test_set, rerank_lists, FLAGS.test_dir, model.hparams.reverse_input, 'test')

	return


def main(_):
	if FLAGS.decode:
		decode()
	else:
		train()

if __name__ == "__main__":
	tf.app.run()
