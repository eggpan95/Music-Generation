import tensorflow as tf
import wave
import numpy
import os
# from scipy.io import wavfile 
# from sklearn.preprocessing import normalize
# import matplotlib.pyplot as plt




class song(object):
	
	def __init__(self, ):
		self.data =numpy.zeros([1,1])
		print(self.data)
		self.load_music()
		self.lastplace = 1001


	def load_music(self):
		print("loading music ... ")
		for filename in os.listdir(os.getcwd()):
			extension = filename.split(".")[-1]
			if extension == "wav":
				print("reading file - " + filename)
				fp = wave.open(filename)
				nchan = fp.getnchannels()
				# print(nchan)
				N = fp.getnframes()
				# print(N)
				dstr = fp.readframes(N*nchan)
				newdata = numpy.fromstring(dstr, numpy.int16)
				newD = numpy.reshape(newdata, (-1,nchan-1))
				# normalized = (newD-min(newD))/(max(newD)-min(newD))
				# self.data = numpy.append(self.data,normalized)
				self.data = newD[:int(newD.shape[0]/10)]
				print("done ..  present size = " + str(self.data.shape[0]))

	def next_batch(self,batch_size):
		start = self.lastplace - batch_size - 1
		# batch_y = self.data[self.lastplace]
		batch_x = self.data[start:self.lastplace-1]
		self.lastplace = self.lastplace + 1
		return numpy.reshape(batch_x ,(1000,1))

	def getlength(self):
		return self.data.shape[0]

train = False
test = True
learning_rate = 0.0015
training_epochs = 100
batch_size = 1000



n_hidden_1 = 500 
n_hidden_2 = 10 
 
n_input = 1000 


X = tf.placeholder("float", [n_input,1])


def encoder(x):
	
	# layer_1 = tf.add(tf.tensordot(weights['encoder_h1'],x,1), biases['encoder_b1'])
	# layer_1 = tf.nn.relu(layer_1)
	layer_1 = tf.layers.dense(inputs=x, units=500, activation=tf.nn.relu)
	layer_2 = tf.layers.dense(inputs=layer_1, units=10, activation=tf.nn.relu)
	# layer_2 = tf.add(tf.tensordot(weights['encoder_h2'],layer_1,1), biases['encoder_b2'])
	# layer_2 = tf.nn.relu(layer_2)
	
	return layer_2

def decoder(x):
	
	# layer_1 = tf.add(tf.tensordot(weights['decoder_h1'],x,1), biases['decoder_b1'])
	# layer_1 = tf.nn.relu(layer_1)

	# layer_2 = tf.add(tf.tensordot(weights['decoder_h2'],layer_1,1), biases['decoder_b2'])
	# layer_2 = tf.nn.relu(layer_2)
	layer_1 = tf.layers.dense(inputs=x, units=500, activation=tf.nn.relu)
	layer_2 = tf.layers.dense(inputs=layer_1, units=1000, activation=tf.nn.relu)
	return layer_2


def _decoder(o):
	layer_1 = tf.layers.dense(inputs=o, units=500, activation=tf.nn.relu)
	layer_2 = tf.layers.dense(inputs=layer_1, units=1000, activation=tf.nn.relu)
	return layer_2

# def decoderhelper(o,sess):
#     print(weights['decoder_h1'].eval(sess).shape)
#     layer_1 = numpy.dot(weights['decoder_h1'].eval(sess),o)+ biases['decoder_b1'].eval(sess)
#     # layer_1 = tf.nn.relu(layer_1)
#     print(layer_1.shape,o.shape,"------------------------------------------------------------------------++++helloHELL")

#     layer_2 = numpy.matrix.dot(weights['decoder_h2'].eval(sess),layer_1)+ biases['decoder_b2'].eval(sess)
#     print(layer_2,"------------------------------------------------------------------------++++helloHELL")

#     # layer_2 = tf.nn.relu(layer_2)
#     print(layer_2,"------------------------------------------------------------------------++++helloHELL")
   
#     return layer_2


# # Store layers weight & bias
# weights = {
#     'encoder_h1': tf.Variable(tf.random_normal([n_hidden_1,n_input ])),
#     'encoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
#     'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'decoder_h2': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
# }
# biases = {
#     'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'decoder_b2': tf.Variable(tf.random_normal([n_input])),
# }


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
pred = decoder_op
# predhelp = tf.py_func(decoderhelper,[o],tf.float32)
o = tf.placeholder("float32", [10,1])
_decoder_op = decoder(o)
_pred = _decoder_op

# Define loss and optimizer
cost = tf.reduce_sum(tf.pow(pred-X, 2))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables

saver = tf.train.Saver()


if train:
	init = tf.global_variables_initializer()
	mysongs = song()
	with tf.Session() as sess:
		sess.run(init)
		print("initialised")

		# Training cycle
		for epoch in range(training_epochs):
			print(epoch)
			avg_cost = 0.
			total_batch = int(mysongs.getlength()/batch_size)
			print(total_batch)
			# Loop over all batches
			for i in range(total_batch):            
				batch_x = mysongs.next_batch(batch_size)
				# print(batch_x.shape,i)
				# batch_y = numpy.array([batch_y])
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c = sess.run([optimizer, cost], feed_dict={X: batch_x})
				# Compute average loss

				avg_cost += c / total_batch
				print(total_batch-i,avg_cost)
			# Display logs per epoch step
			if epoch % 1 == 0:
				print ("Epoch:", '%04d' % (epoch+1), "cost=", \
					"{:.9f}".format(avg_cost))
				save_path = saver.save(sess, "/Documents/tushhar/2xt/2xt/projects/tensorflow/deep learning/Music/tmp/model.ckpt")
				print("Model saved in file: %s" % save_path)
		print ("Optimization Finished!")

		# wavfile.write("/Documents/tushhar/2xt/2xt/projects/tensorflow/deep learning/Music/tmp/predictedsong.wav",,denormalised)
		# save_path = saver.save(sess, "/Documents/tushhar/2xt/2xt/projects/tensorflow/deep learning/Music/tmp/model.ckpt")
		# print("Model saved in file: %s" % save_path)
		# plt.show()




	# Test model
	# correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	# Calculate accuracy
	# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	# print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


if test:
	testsong = numpy.zeros([1,1])

	with tf.Session() as sess:
		saver.restore(sess, "/Documents/tushhar/2xt/2xt/projects/tensorflow/deep learning/Music/tmp/model.ckpt")
		print("Model restored.")
		writer = tf.summary.FileWriter('./graphs',sess.graph)
		for k in range(100):
			# session=sess, feed_dict={x: tf.random_normal([10, 1])}
			# value = decoderhelper(numpy.matrix(numpy.random.rand(10, 1)),sess)
			value = sess.run(_pred,feed_dict={o: numpy.matrix(numpy.random.rand(10, 1))})
			print(value.dtype,value.shape)
			testsong = numpy.append(testsong,value)
			print(testsong.shape)
	writer.close()        



	wf = wave.open('song.wav','w')
	wf.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
	print(song.dtype)
	# song =song.astype(np.int16)
	wf.writeframesraw(song)
	wf.close()






#   TODO try with helper function 