import cPickle as pickle
import numpy as np
import ujson as json
import os

data_dir = './data/'
embeddings_dir = os.path.join(data_dir, 'glove.6B.50d.txt')
train_dir = 'train_set.json'

def prepare_embeds(dir=embeddings_dir):
	try:

		with open(os.path.join(data_dir, "embeds.pkl"), 'rb') as f:
			embeds = pickle.load(f)
		print "embeddings exist."
		vocab = embeds["vocab"]
		vectors = embeds["vectors"]
		print "returning vectos and vocab."
		return vocab, vectors
	except:
		print "creating embeddings."
		with open(dir) as f:
			x = f.readlines()
		vocab = []
		vectors = []
		for i in x:
			line = i.strip('\n').split(' ')
			if line[0]=="unk":
				vocab.insert(0, "unk")
				vectors.insert(0, map(np.float32, line[1:]))
				continue
			vocab.append(line[0])
			vectors.append(map(np.float32, line[1:]))
		with open(os.path.join(data_dir, 'embeds.pkl'), 'wb') as f:
			pickle.dump({'vocab':vocab, 'vectors':vectors}, f)
		print "done creating embeddings."
		return vocab, vectors

def create_intent_list(data_dir=train_dir):
	with open(data_dir) as f:
		data = json.load(f)
	return list({i['intent'] for i in data})

def create_entities_list(data_dir=train_dir):
	with open(data_dir) as f:
		data = json.load(f)
	return list({j['entity'] for i in data for j in i['entities']})

def get_entity_k_hot(entities, entity_list):
	k_hot = [[0. for _ in xrange(len(entity_list))]]
	for i in entities:
		idx = entity_list.index(i)
		k_hot[0][idx] = 1.
	return k_hot

def get_intents_one_hot(intent, intent_list):
	one_hot = [[1. if i==intent else 0. for i in intent_list]]
	return one_hot

def prepare_train_set(dir=train_dir):
	try:
		with open('./data/prepared_train_set.pkl', 'rb') as f:
			examples = pickle.load(f)
	except:
		with open(dir) as f:
			data = json.load(f)
		entity_list = create_entities_list()
		intent_list = create_intent_list()
		examples = []
		for i in data:
			example = {}
			example["text"] = np.array([i["text"].split()])
			entities = [j["entity"] for j in i["entities"]]
			example["entities"] = np.array(get_entity_k_hot(entities, entity_list))
			example["targets"] = np.array(get_intents_one_hot(i["intent"], intent_list))
			examples.append(example)
		print examples
		with open('./data/prepared_train_set.pkl', 'wb') as f:
			pickle.dump(examples, f)
	finally:
		return examples


if __name__=="__main__":
	prepare_train_set()