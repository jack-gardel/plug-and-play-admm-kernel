import numpy as np
import matplotlib.pyplot as plt

# Used to produce figures for Gibb's phenomenon
if __name__ == "__main__":
  W = np.pi/2
  N = 10
  sinc_small = np.array([(W/np.pi)*np.sinc(W*(n-N/2)/np.pi) for n in range(N)])
  N = 100
  sinc_med = np.array([(W/np.pi)*np.sinc(W*(n-N/2)/np.pi) for n in range(N)])
  N = 1000
  sinc_large = np.array([(W/np.pi)*np.sinc(W*(n-N/2)/np.pi) for n in range(N)])

  plt.figure(1)
  plt.title("FT of sinc signal of length 10")
  plt.xlabel("Frequency (radians)")
  plt.plot(np.linspace(0,2*np.pi,4096), np.abs(np.fft.fft(sinc_small, 4096)))
  plt.savefig("gibbs_1.jpg")

  plt.figure(2)
  plt.title("FT of sinc signal of length 100")
  plt.xlabel("Frequency (radians)")
  plt.plot(np.linspace(0,2*np.pi,4096), np.abs(np.fft.fft(sinc_med, 4096)))
  plt.savefig("gibbs_2.jpg")

  plt.figure(3)
  plt.title("FT of sinc signal of length 1000")
  plt.xlabel("Frequency (radians)")
  plt.plot(np.linspace(0,2*np.pi,4096), np.abs(np.fft.fft(sinc_large, 4096)))
  plt.savefig("gibbs_3.jpg")

  plt.figure(4)
  plt.title("FT of sinc signal of length 1000 with a Hamming Window")
  plt.xlabel("Frequency (radians)")
  plt.plot(np.linspace(0,2*np.pi,4096), np.abs(np.fft.fft(np.multiply(sinc_large, np.hamming(N)), 4096)))
  plt.savefig("hamming.jpg")