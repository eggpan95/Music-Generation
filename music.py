import wave
import numpy
import matplotlib.pyplot as plt

fp = wave.open("05. Time (From 'Inception').wav")
nchan = fp.getnchannels()
print(nchan)
N = fp.getnframes()
print(fp.getparams())
dstr = fp.readframes(N*nchan)
fp.close()
data = numpy.fromstring(dstr, numpy.int16)
data = numpy.reshape(data, (-1,nchan-1))
min1 = min(data)
max1 = max(data)
data1 = data[0:2600000]
norm = (data1 - min1)/(max1-min1)
# norm1 = data1 / numpy.linalg.norm(data)
# print(data)
# plt.plot(data,'r--')
# plt.plot(norm1,'k')
# plt.show()
wf = wave.open('song.wav','w')
wf.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
# renormalised = norm1*numpy.linalg.norm(data)
renormalised = norm *(max1-min1) +min1
print(renormalised.dtype,data.dtype)
print(renormalised)
print(data1)
renormalised =renormalised.astype(numpy.int16)
wf.writeframesraw(renormalised)
wf.close()

# testsong = song()
#     save_song = numpy.zeros([1,1])
#     for i in range(total_batch):            
#         batch_x, batch_y = mysongs.next_batch(batch_size)
#         predictions = sess.run(pred, feed_dict={x: batch_x})
#         print(predictions)
#         save_song = numpy.append(save_song,predictions)
# #     plt.plot(plot)
#     denormalised = save_song*(max(save_song)-min(save_song))+min(save_song)
#     wf = wave.open("C/Documents/tushhar/2xt/2xt/projects/tensorflow/deep learning/Music/tmp/predictedsong.wav")
#     wf.writeframes(denormalised)