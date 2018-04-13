# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import json
import random
import os

class Raw_data:
	def __init__(self, data_path = None, file_prefix = None):
		if data_path == None:
			self.embed_size = -1
			self.rank_list_size = -1
			self.features = []
			self.dids = []
			self.initial_list = []
			self.qids = []
			self.gold_list = []
			self.gold_weights = []
			return

		settings = json.load(open(data_path + 'settings.json'))
		self.embed_size = settings['embed_size']
		self.rank_list_size = settings['rank_cutoff']

		self.features = []
		self.dids = []
		feature_fin = open(data_path + file_prefix + '/' + file_prefix + '.feature')
		for line in feature_fin:
			arr = line.strip().split(' ')
			self.dids.append(arr[0])
			self.features.append([0.0 for _ in xrange(self.embed_size)])
			for x in arr[1:]:
				arr2 = x.split(':')
				self.features[-1][int(arr2[0])] = float(arr2[1])
		feature_fin.close()

		self.initial_list = []
		self.qids = []
		init_list_fin = open(data_path + file_prefix + '/' + file_prefix + '.init_list')
		for line in init_list_fin:
			arr = line.strip().split(' ')
			self.qids.append(arr[0])
			self.initial_list.append([int(x) for x in arr[1:]])
		init_list_fin.close()

		self.gold_list = []
		gold_list_fin = open(data_path + file_prefix + '/' + file_prefix + '.gold_list')
		for line in gold_list_fin:
			self.gold_list.append([int(x) for x in line.strip().split(' ')[1:]])
		gold_list_fin.close()

		self.gold_weights = []
		gold_weight_fin = open(data_path + file_prefix + '/' + file_prefix + '.weights')
		for line in gold_weight_fin:
			self.gold_weights.append([float(x) for x in line.strip().split(' ')[1:]])
		gold_weight_fin.close()

		self.initial_scores = []
		#if os.path.isfile(data_path + file_prefix + '/' + file_prefix + '.intial_scores'):
		with open(data_path + file_prefix + '/' + file_prefix + '.initial_scores') as fin:
			for line in fin:
				self.initial_scores.append([float(x) for x in line.strip().split(' ')[1:]])

	def pad(self, rank_list_size, reverse_input):
		self.rank_list_size = rank_list_size
		self.features.append([0 for _ in xrange(self.embed_size)])  # vector for pad
		for i in xrange(len(self.initial_list)):
			if len(self.initial_list[i]) < self.rank_list_size:
				if reverse_input:
					self.initial_list[i] += [-1] * (self.rank_list_size - len(self.initial_list[i]))
				else:
					self.initial_list[i] = [-1] * (self.rank_list_size - len(self.initial_list[i])) + self.initial_list[i]
				self.gold_list[i] += [-1] * (self.rank_list_size - len(self.gold_list[i]))
				self.gold_weights[i] += [0.0] * (self.rank_list_size - len(self.gold_weights[i]))
				self.initial_scores[i] += [0.0] * (self.rank_list_size - len(self.initial_scores[i]))

	def boost_training_data(self, max_boosted_num, swap_number):
		boosted_initial_list = []
		boosted_gold_list = []
		boosted_gold_weights = []
		for i in xrange(len(self.initial_list)):
			new_gold_list = list(self.gold_list[i])
			list_length = len(new_gold_list)
			#swap for swap_number times
			for _ in xrange(max_boosted_num):
				list_change = False
				for j in xrange(swap_number):
					_1 = int(random.random()*(list_length))
					_2 = int(random.random()*(list_length))
					if new_gold_list[_1] == new_gold_list[_2]:
						continue
					if self.gold_weights[i][new_gold_list[_1]] == self.gold_weights[i][new_gold_list[_2]]:
						list_change = True
						tmp = new_gold_list[_1]
						new_gold_list[_1] = new_gold_list[_2]
						new_gold_list[_2] = tmp
				if list_change:
					boosted_initial_list.append(list(self.initial_list[i]))
					boosted_gold_list.append(list(new_gold_list))
					boosted_gold_weights.append(list(self.gold_weights[i]))

		self.initial_list += boosted_initial_list
		self.gold_list += boosted_gold_list
		self.gold_weights += boosted_gold_weights


def read_data(data_path, file_prefix):
	data = Raw_data(data_path, file_prefix)
	return data

def generate_ranklist(data, rerank_lists, reverse_input):
	if len(rerank_lists) != len(data.initial_list):
		raise ValueError("Rerank ranklists number must be equal to the initial list,"
						 " %d != %d." % (len(rerank_lists)), len(data.initial_list))
	qid_list_map = {}
	for i in xrange(len(data.qids)):
		if len(rerank_lists[i]) != len(data.initial_list[i]):
			raise ValueError("Rerank ranklists length must be equal to the gold list,"
							 " %d != %d." % (len(rerank_lists[i]), len(data.initial_list[i])))
		#remove duplicate and organize rerank list
		index_list = []
		index_set = set()
		for j in rerank_lists[i]:
			idx = len(rerank_lists[i]) - 1 - j if reverse_input else j
			if idx not in index_set:
				index_set.add(idx)
				index_list.append(idx)
		for idx in xrange(len(rerank_lists[i])):
			if idx not in index_set:
				index_list.append(idx)
		#get new ranking list
		qid = data.qids[i]
		did_list = []
		new_list = [data.initial_list[i][idx] for idx in index_list]
		for ni in new_list:
			if ni >= 0:
				did_list.append(data.dids[ni])
		qid_list_map[qid] = did_list
	return qid_list_map

def output_ranklist(data, rerank_lists, output_path, reverse_input, file_name = 'test'):
	qid_list_map = generate_ranklist(data, rerank_lists, reverse_input)
	fout = open(output_path + file_name + '.ranklist','w')
	for qid in data.qids:
		for i in xrange(len(qid_list_map[qid])):
			fout.write(qid + ' Q0 ' + qid_list_map[qid][i] + ' ' + str(i+1)
							+ ' ' + str(0-i) + ' RankLSTM\n')
	fout.close()

