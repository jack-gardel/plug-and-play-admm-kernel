import numpy as np
import pdb

def dbl_circ(kernel: np.ndarray, input_shape: tuple[int, int]):
  N = input_shape[0] * input_shape[1]
  dim_k = kernel.shape
  h = kernel
  # Create the doubly-block circulant matrix
  H = np.zeros(((input_shape[0]-dim_k[0]+1)*(input_shape[1]-dim_k[1]+1), N))
  h_flat = h.flatten()
  h_row = []
  i = 0
  while i < dim_k[0]*dim_k[1]:
    if i % dim_k[1] == 0 and i != 0:
      h_row += (input_shape[1] - dim_k[1])*[0]
      h_row.append(h_flat[i])
      i += 1
    else:
      h_row.append(h_flat[i])
      i += 1
  offset = 0
  for i in range(H.shape[0]):
    for j in range(H.shape[1]):
      if j >= offset and j < offset + len(h_row):
        H[i,j] = h_row[j-offset]
    if i % (input_shape[1]-dim_k[1]+1) == input_shape[1]-dim_k[1] and i != 0:
      offset += dim_k[1]
    else:
      offset += 1

  return H

if __name__ == "__main__":
  high_res = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
  N = high_res.shape[0] * high_res.shape[1]
  # Produce the blur kernel from a gaussian
  dim_k = 2
  h = np.array([[1,0],[0,1]])
  # Create the doubly-block circulant matrix
  H = np.zeros(((high_res.shape[0]-dim_k+1)*(high_res.shape[1]-dim_k+1), N))
  h_flat = h.flatten()
  h_row = []
  i = 0
  while i < dim_k*dim_k:
    if i % dim_k == 0 and i != 0:
      h_row += (high_res.shape[1] - dim_k)*[0]
      h_row.append(h_flat[i])
      i += 1
    else:
      h_row.append(h_flat[i])
      i += 1
  offset = 0
  for i in range(H.shape[0]):
    print(f"{i}: {offset}")
    for j in range(H.shape[1]):
      if j >= offset and j < offset + len(h_row):
        H[i,j] = h_row[j-offset]
    if i % (high_res.shape[1]-dim_k+1) == high_res.shape[1]-dim_k and i != 0:
      offset += dim_k
    else:
      offset += 1

  print(h_row)
  print(H)
  print(H@high_res.flatten())
