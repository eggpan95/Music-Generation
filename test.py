import tensorflow as tf
import wave
import numpy as np




# x = tf.placeholder("float", [n_hidden_2,1])






# pred = decoder_op


saver = tf.train.Saver()
testsong = np.zeros([1,1])



with tf.Session() as sess:
	saver.restore(sess, "C:/Users/Cooridinate home/Documents/tushhar/2xt/2xt/projects/tensorflow/deep learning/Music/tmp/model.ckpt")
	print("Model restored.")
	for k in range(10):
		# session=sess, feed_dict={x: tf.random_normal([10, 1])}
		value = sess.run(pred,feed_dict={x: tf.random_normal([10, 1])})
		print(value.dtype,value.shape)
		testsong = np.append(testsong,value)
		print(testsong.shape)



wf = wave.open('song.wav','w')
wf.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
print(song.dtype)
# song =song.astype(np.int16)
wf.writeframesraw(song)
wf.close()