import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import scipy.linalg as lsp
#import cv2
#from bm3d import bm3d
import pdb
from skimage import data, color
import skimage.transform as imt
from skimage.restoration import denoise_tv_chambolle

def make_gauss(min_idx,max_idx):
  delta = 0.25
  sigma = np.array([[2,1],[1,2]])
  mu = np.array([0,0])
  x = np.arange(min_idx, max_idx,delta)
  y = np.arange(min_idx, max_idx,delta)
  f_X = np.zeros((len(y), len(x)))
  for i in range(len(y)):
    for j in range(len(x)):
      g = (2/3)*(x[j])**2 + (2/3)*(y[i])**2 - (2/3)*(x[j])*(y[i])
      f_X[i,j] = (1/(2*np.pi*np.sqrt(3)))*np.exp(-0.5*g)
  return f_X

def make_observed(img, Ksq):
  observed_img = imt.downscale_local_mean(img, (Ksq, Ksq))
  for a in observed_img:
    a += np.random.normal(0,0.1)
  return observed_img
  
  

def denoise(a, shape, sigma):
  smoothed_a = denoise_tv_chambolle(a.reshape(shape), weight=0.2)
  return smoothed_a.reshape((-1,1))

#high_res = plt.imread("data/lotus.jpg")
#low_res = plt.imread("data/lotus_low.jpg")
K = 4 # Downsampling factor
Ksq = 2 # square root of K
high_res = make_gauss(-5,5)
low_res = make_observed(high_res, Ksq)
#pdb.set_trace()

# N = total pixels in high-res image
# M = total pixels in low-res image
N = high_res.shape[0] * high_res.shape[1]
M = low_res.shape[0] * low_res.shape[1]

# Produce the blur kernel from a gaussian
h = np.array([[(1/(2*np.pi))*np.exp(-0.5*((i-4)**2 + (j-4)**2))  for i in range(7)] for j in range(7)])
h_tilde = np.fft.ifft2(np.fft.fft2(h)*np.conj(np.fft.fft2(h)))
h_0_tilde = np.array([h_tilde[i//K] for i in range(h_tilde.shape[0]//K)])
fft_h_0_tilde = np.abs(np.fft.fft2(h_0_tilde))

H = lsp.circulant([sp.norm.pdf(i-N//2) for i in range(N)])

# Produce S as a K-fold downsampling operator
S = np.array([[1 if i == K*j else 0 for i in range(N)] for j in range(M)])

# G combines the functionality of H and S
G = S@H

rho = 1
lmbda = 1 # regularization
eta = 0.8 # change this
gamma = 1 # change this
delta = 0
x = np.zeros((high_res.shape))
v = np.zeros((high_res.shape))
u = np.zeros((high_res.shape))
y = low_res
iterations = 20
# The algorithm
gtg_inv = np.linalg.inv(G.T@G + rho*np.identity(N))
#pdb.set_trace()
for i in range(iterations):
  print(i)
  x_old = x
  v_old = v
  u_old = u
  delta_old = delta
  x_tilde = v-u
  b = G.T@y + rho*x_tilde
  x = b/rho - (1/rho)*G.T@(np.fft.irfft(np.divide(np.fft.fft(G@b),(fft_h_0_tilde+rho))))#gtg_inv@(G.T@y + rho*(v-(1/rho)*u))
  v = denoise(x+u, high_res.shape, np.sqrt(lmbda/rho))
  u += x - v
  delta = (1/np.sqrt(N))*(np.sqrt((x-x_old).T@(x-x_old)) + np.sqrt((v-v_old).T@(v-v_old)) + np.sqrt((u-u_old).T@(u-u_old)))
  #if delta >= eta*delta_old:
  rho = gamma*rho

  X = np.reshape(x, (high_res.shape[0], high_res.shape[1]))
pdb.set_trace()


