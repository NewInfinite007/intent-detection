import tensorflow as tf
import model
import misc

saver = tf.train.Saver()
entity_list = misc.create_entities_list()
intent_list = misc.create_intent_list()

with tf.Session() as sess:
	sess.run(model.init_op)
	saver.restore(sess, tf.train.latest_checkpoint('./models'))

	while raw_input("would you like to continue? :")=='c':
		input_string = raw_input("enter input :")
		input_vec = [input_string.split()]
		entity = raw_input("enter entity :")
		entity_vec = misc.get_entity_k_hot([entity], entity_list)
		feed_dict = {
			model.dense_rep:input_vec,
			model.sparse_rep:entity_vec
		}
		preds = sess.run(model.predictions, feed_dict=feed_dict)
		print "prediction :", preds