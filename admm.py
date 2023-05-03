import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from dbl_circ import dbl_circ
from downsample import downsample
import cvxpy as cp
import sys

FIGS = True

# Take an image vector and construct an image
def unflatten(img_vec: np.ndarray, output_shape: tuple[int, int]):
  I = np.zeros(output_shape)
  for i in range(output_shape[0]):
    I[i] = img_vec[output_shape[1]*(i):output_shape[1]*(i+1)]
  
  return I

# Create an image with a Gaussian shape
def make_gauss(min_idx: int, max_idx: int):
  delta = 0.25
  x = np.arange(min_idx, max_idx, delta)
  y = np.arange(min_idx, max_idx, delta)
  f_X = np.zeros((len(y), len(x)))
  for i in range(len(y)):
    for j in range(len(x)):
      g = (2/3)*(x[j])**2 + (2/3)*(y[i])**2 - (2/3)*(x[j])*(y[i])
      f_X[i,j] = (1/(2*np.pi*np.sqrt(3)))*np.exp(-0.5*g)
  return f_X

# Truncate and add noise
def make_observed(img: np.ndarray, K: int, L: int, H: int):
  truncated = img[H//2:img.shape[0]-H+1+H//2, L//2:img.shape[1]-L+1+L//2]
  observed = np.zeros((truncated.shape[0]//K, truncated.shape[1]//K))
  for i in range(observed.shape[0]):
    for j in range(observed.shape[1]):
      observed[i,j] = truncated[i*K,j*K] + (np.random.normal(size=1) / 100)
  
  return observed

# Denoise operation
def denoise(a, shape, sigma):
  smoothed_a = denoise_tv_chambolle(a.reshape(shape), weight=sigma)
  return smoothed_a

# Update step for x
def update_x(rho, G, x_tilde, y):
  output_shape = x_tilde.shape
  # Flatten images
  x_tilde = x_tilde.flatten()
  x_tilde = x_tilde.reshape(-1,1)
  y = y.flatten()
  y = y.reshape(-1,1)
  # Solve minimization problem
  x = cp.Variable((G.shape[1],1))
  objective = cp.Minimize(cp.sum_squares(G@x-y) + 0.5*rho*cp.sum_squares(x-x_tilde))
  prob = cp.Problem(objective)
  optimal = prob.solve()
  return x.value.reshape(output_shape)

def admm(dim_k, kernel):
  K = 2 # Downsampling factor
  high_res = plt.imread("data/lotus.jpg")

  kernel = "hamming"
  # Produce the blur kernel
  if kernel == "gauss":
    # Gaussian kernel
    h = np.array([[1 for i in range(dim_k)] for i in range(dim_k)])
    h = np.array([[(1/(2*np.pi))*np.exp(-0.5*((i-(dim_k//2+1))**2 + (j-(dim_k//2+1))**2))  for i in range(dim_k)] for j in range(dim_k)])
  elif kernel == "sinc":
    # Sinc kernel
    W = np.pi/K
    x_1 = np.array([i-dim_k//2 for i in range(dim_k)])
    x_2 = np.array([i-dim_k//2 for i in range(dim_k)])
    h = np.array([[((W**2)/(np.pi**2))*np.sinc(W*x_1[i]/np.pi)*np.sinc(W*x_2[j]/np.pi) for i in range(dim_k)] for j in range(dim_k)])
  elif kernel == "hamming":
    # Sinc kernel with a hamming window
    W = np.pi/K
    x_1 = np.array([i-dim_k//2 for i in range(dim_k)])
    x_2 = np.array([i-dim_k//2 for i in range(dim_k)])
    ham = np.array([np.hamming(dim_k)])
    ham = np.sqrt(np.outer(ham,ham))
    h = np.array([[((W**2)/(np.pi**2))*np.sinc(W*x_1[i]/np.pi)*np.sinc(W*x_2[j]/np.pi) for i in range(dim_k)] for j in range(dim_k)])
    h = np.multiply(ham,h)
  # Create the doubly-block circulant matrix
  H = dbl_circ(h, high_res.shape)

  # Produce S as a K-fold downsampling operator
  filtered_shape = (high_res.shape[0] - dim_k + 1, high_res.shape[1] - dim_k + 1)
  S = downsample(filtered_shape, K)
  y = make_observed(high_res, K, dim_k, dim_k)
  G = S@H

  rho = 2
  lmbda = 0.5
  eta = 0.8
  gamma = 1
  delta = 0

  x = np.zeros(high_res.shape)
  v = np.zeros(high_res.shape)
  u = np.zeros(high_res.shape)

  iterations = 20
  loss = []
  # ADMM steps
  for i in range(iterations):
    print(f"\r{i}", end="")
    x_old = x
    v_old = v
    u_old = u
    delta_old = delta
    x_tilde = v-u
    # Step for x_k+1
    x = update_x(rho, G, x_tilde, y)
    # Step for v_k+1
    v = denoise(x+u, high_res.shape, np.sqrt(lmbda/rho))
    # Step for u_k+1
    u += x - v
    delta = (1/np.sqrt(x.shape[0]*x.shape[1]))*(np.sqrt((x.flatten()-x_old.flatten()).T@(x.flatten()-x_old.flatten())) + np.sqrt((v.flatten()-v_old.flatten()).T@(v.flatten()-v_old.flatten())) + np.sqrt((u.flatten()-u_old.flatten()).T@(u.flatten()-u_old.flatten())))
    if delta >= eta*delta_old:
      rho = gamma*rho
    # Truncate so that transient points in the image do not get taken into account for the loss
    x_truncated = x[dim_k//2:x.shape[0]-dim_k//2,dim_k//2:x.shape[1]-dim_k//2]
    high_res_truncated = high_res[dim_k//2:high_res.shape[0]-dim_k//2,dim_k//2:high_res.shape[1]-dim_k//2]
    # Append loss
    loss.append((x_truncated.flatten()-high_res_truncated.flatten()).T@(x_truncated.flatten()-high_res_truncated.flatten()))

  if FIGS:
    print("")
    plt.figure()
    plt.subplot(221)
    plt.plot(loss)
    plt.subplot(222)
    plt.imshow(y)
    plt.subplot(223)
    plt.imshow(x)
    plt.subplot(224)
    plt.imshow(high_res)
    plt.savefig(f"loss_and_img_{dim_k}_{kernel}.jpg")

    plt.imsave(f"data/sinc_img_{dim_k}_{kernel}.jpg", x)

  return loss[-1]

if __name__ == "__main__":
  sizes = [3,5,7,9,13,15]
  loss = []
  try:
    kernel = sys.argv[1]
  except:
    kernel = "hamming"
  for s in sizes:
    loss.append(admm(s, kernel))
  print()
  
  plt.figure()
  plt.plot(sizes,loss)
  plt.title("Loss vs Kernel Size")
  plt.xlabel("Kernel Size (both dimensions)")
  plt.ylabel("Loss")
  plt.show()

