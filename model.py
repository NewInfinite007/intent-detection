import tensorflow as tf
import misc
import os


dense_rep_shape = 50
sparse_rep_shape = len(misc.create_entities_list())
n_epochs = 100
lr = 0.001

vocab, vectors = misc.prepare_embeds()
print "got vectors."
num_intents = len(misc.create_intent_list())


dense_rep = tf.placeholder(dtype=tf.string)
sparse_rep = tf.placeholder(dtype=tf.float32)
targets = tf.placeholder(dtype=tf.float32)
embeddings = tf.Variable(vectors)

dense_rep_indices = tf.contrib.lookup.string_to_index(dense_rep, vocab, default_value=0)
dense_rep_embeddings = tf.nn.embedding_lookup(embeddings, dense_rep_indices)

#dense processing
dense_rep_averaged = tf.reduce_mean(dense_rep_embeddings, axis=1)

#mlp layers
W1 = tf.Variable(tf.random_uniform([dense_rep_shape, dense_rep_shape], -1., 1.))
b1 = tf.Variable(tf.ones([dense_rep_shape], dtype=tf.float32))
o1 = tf.nn.relu(tf.add(tf.matmul(dense_rep_averaged, W1), b1))

#concatenating dense and sparse output
final_input = tf.concat([o1, sparse_rep], axis=1)
#
# #final layer
final_shape_rep = dense_rep_shape+sparse_rep_shape
final_W = tf.Variable(tf.random_uniform([final_shape_rep, num_intents], -1.,1.))
final_b = tf.Variable(tf.ones([num_intents], dtype=tf.float32))
logits = tf.add(tf.matmul(final_input, final_W), final_b)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
#
# #backprop
optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(loss)
saver = tf.train.Saver()

init_op = [tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()]

with tf.Session() as sess:
	sess.run(init_op)
	if os.path.exists('./models/checkpoint'):
		saver.restore(sess, './models/model.ckpt')
	global_step = 0
	for epoch in xrange(n_epochs):
		print "current epoch :", epoch
		overall_loss = 0.
		count = 0
		accuracy = 0.
		for i in misc.prepare_train_set():
			text_tokens = i["text"]
			entities = i["entities"]
			target = i["targets"]
			count +=1
			local_loss,_ = sess.run([loss, train_op], feed_dict={
												dense_rep:text_tokens,
												sparse_rep:entities,
												targets:target})
			global_step+=1
			overall_loss+=local_loss
		print "loss per epoch :", overall_loss/count
		saver.save(sess, "./models/model.ckpt", global_step)