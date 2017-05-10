from keras.layers import Input, Dense
from keras.models import Model
import wave
import numpy as np
import os
import matplotlib.pyplot as plt


class song(object):
	
	def __init__(self, ):
		self.data =np.zeros([1,1])
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
				newdata = np.fromstring(dstr, np.int16)
				newD = np.reshape(newdata, (-1,nchan-1))
				normalized = (newD-min(newD))/(max(newD)-min(newD))
				self.data = np.append(self.data,normalized)
				# self.data = (newD[:int(newD.shape[0])])
				# plt.plot(self.data)
				# self.data = self.mu_law_encode(self.data)
				# plt.plot(self.data)
				# plt.show()
				print("done ..  present size = " + str(self.data.shape[0]))

	def next_batch(self):
		start = self.lastplace - 1000 - 1
		# batch_y = self.data[self.lastplace]
		batch_x = self.data[start:self.lastplace-1]
		self.lastplace = self.lastplace + 1
		return np.reshape(batch_x ,(1000,1))

	def getlength(self):
		return self.data.shape[0]

	def mu_law_encode(self,audio):
		mu = 255
		safe_audio_abs=np.minimum(np.abs(audio),1.0)
		magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
		signal = np.sign(audio) * magnitude
		return ((signal + 1) / 2 * mu + 0.5).astype(np.int32)
	def mu_law_decode(self ,output):
		mu = 255
		signal = 2*(output.astype(np.float)/mu)-1
		magnitude = (1/mu)*((1 + mu)**np.abs(signal)-1)
		return np.sign(signal)*magnitude
# this is the size of our encoded representations
encoding_dim = 50  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(1000,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(1000, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)


# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


mysongs = song()
total_batch = int(mysongs.getlength()/1000)
print(total_batch)
x_train = np.zeros([1,1000])
for i in range(total_batch):
	next_batch = np.transpose(mysongs.next_batch())
	x_train = np.append(x_train,next_batch,axis=0)
	print (x_train.shape ,"----------------------------------------<")


autoencoder.fit(x_train, x_train,epochs=5,shuffle=True)
autoencoder.save('model.h5')
decoded_song = np.zeros([1,1])

for k in range(1000):
	value = decoder.predict(np.matrix(np.random.rand(1, 50)))
	# print(value.dtype,value.shape)
	decoded_song = np.append(decoded_song,value)
print(decoded_song.shape)

wf = wave.open('song.wav','w')
wf.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
print(decoded_song.dtype)
decoded_song =decoded_song.astype(np.int16)
wf.writeframesraw(decoded_song)
wf.close()
plt.plot(decoded_song)
plt.show()

# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)



# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()