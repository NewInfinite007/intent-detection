from bottle import route, run, template, post, request
import misc
import model
import ujson as json
import tensorflow as tf

saver = tf.train.Saver()
entity_list = misc.create_entities_list()
intent_list = misc.create_intent_list()
sess = tf.Session()
sess.run(model.init_op)
saver.restore(sess, tf.train.latest_checkpoint('./models'))

@route('/')
def index():
	return template(open('index.html', 'r').read())

@route('/train')
def train():
	return None

@post('/predict')
def predict():
	postdata = json.loads(request.body.read())
	input_feed = postdata['input']
	input_vec = [input_feed.split()]
	entity_feed = postdata['entity']
	entity_vec = misc.get_entity_k_hot([entity_feed], entity_list)
	feed_dict = {
		model.dense_rep:input_vec,
		model.sparse_rep:entity_vec
	}
	predictions = sess.run(model.predictions, feed_dict)
	return str(predictions)


run(host='localhost', port=9000)