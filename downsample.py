import numpy as np
import pdb

def downsample(input_shape: tuple[int, int], K: int):
  M = input_shape[0]
  N = input_shape[1]
  S = np.zeros(((M // K) * (N // K), M*N))
  offset = 0
  for i in range((M // K) * (N // K)):
    S[i,offset] = 1
    if i % (N // K) == N // K - 1:
      offset += N*(K-1) + K
    else:
      offset += K
  
  return S

if __name__ == "__main__":
  # input is MxN, downsampled by K
  K = 2
  M = 4
  N = 4

  S = np.zeros(((M // K) * (N // K), M*N))
  offset = 0
  for i in range((M // K) * (N // K)):
    S[i,offset] = 1
    if i % (N // K) == N // K - 1:
      offset += N*(K-1) + K
    else:
      offset += K

  print(S)
  T = downsample((4,4), 2)
  print(T)
  
    