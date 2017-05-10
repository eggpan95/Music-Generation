from pydub import AudioSegment
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile 
from scipy.fftpack import fft

def get_wav(filename):
    extension = filename.split(".")[-1]
    if extension == "wav":
        return filename
    elif extension == "mp3":
        sound = AudioSegment.from_mp3(filename)
        newfilename = filename.replace(extension, "wav")
        sound.export(newfilename, format="wav")
        print("Converted {0} to {1}".format(filename, newfilename))
        return newfilename
def f(filename):
    fs, data = wavfile.read(filename) # load the data
    print(data.dtype)
    data = data//(2.**15)
    s = data[:,0]
    print(s.dtype)
    print("taking fft for " + filename)
    p =fft(s)
    d = len(p)//2  
    plt.plot(abs(p[:(d-1)]),'r')
    print("done preprocess" + filename)

if __name__ == '__main__':
	
	for filename in os.listdir(os.getcwd()):
		# print(get_wav(filename)) #to convert mp3 to wav
		f(filename)
		plt.show()
