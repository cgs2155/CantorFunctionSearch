import numpy as np
from scipy.linalg import eigh_tridiagonal
from collections import Counter

def tree_mag(X:list[int]):
    d = len(X)
    if type(X) == np.ndarray:
      X = X.tolist()
    X = [1]+X
    total = 1

    for i in range(0, d):
        total += np.prod(X[-(i+1):])
        
    return total

def gt_mag(X:list[int]):
    return tree_mag(X) + tree_mag(X[1:])

def direct_sum(A, B):
  """
  Computes the direct sum of two matrices. (Thank you Google)

  Args:
    A: The first matrix (NumPy array).
    B: The second matrix (NumPy array).

  Returns:
    The direct sum of A and B as a NumPy array.
  """
  m, n = A.shape
  p, q = B.shape
  result = np.zeros((m + p, n + q))
  result[:m, :n] = A
  result[m:, n:] = B
  return result

def e_n(n, d):
    # Creates a d-dimensional vector with 1 at the n-th position and 0 elsewhere
    vec = np.zeros(d)
    vec[n] = 1
    return vec

def compute_poly_roots(betas, all=True):
    """
    Compute the roots of the polynomial p_{2n+1}(λ) defined by:
      p_0(λ) = 1,
      p_1(λ) = λ,
      p_{k+1}(λ) = λ p_k(λ) - betas[k-1] * p_{k-1}(λ),
    where betas is a list/array of length 2n.
    """
    m = len(betas)
    # Represent polynomials by their coefficient arrays (highest degree first)
    polys = [np.array([1.]), np.array([1., 0.])]

    p_prev = polys[0]#np.array([1.])
    p_curr = polys[-1]#np.array([1., 0.])  # corresponds to λ
    if m == 0:
        return np.array([0.])
    
    for beta in betas:
        # Multiply p_curr by λ: equivalent to appending a zero at the end.
        # p_prev is of lower degree; pad it with two leading zeros to match the degree of λ*p_curr.
        p_next = np.concatenate((p_curr, [0])) - beta * np.concatenate((np.zeros(2), p_prev))
        #polys.append(p_next)
        p_prev, p_curr = p_curr, p_next
        
    return np.roots(p_curr)


def compute_poly_roots_tridiag(betas):
    """
    Compute the roots of the polynomial defined by:
      p_0(x) = 1,
      p_1(x) = x,
      p_{k+1}(x) = x p_k(x) - beta[k-1]*p_{k-1}(x),
    using the fact that the roots are the eigenvalues of the associated 
    symmetric tridiagonal (Jacobi) matrix.
    
    Parameters:
      betas: array-like of length 2n, defining the recurrence.
      
    Returns:
      roots: the eigenvalues (roots) of the polynomial of degree 2n+1.
    """
    n = len(betas)  # betas length = 2n
    #N = 2 * n + 1  # polynomial degree is 2n+1
    N = n+1
    # The Jacobi matrix has zeros on the diagonal.
    diag = np.zeros(N)
    # The off-diagonal entries are sqrt(betas); note: ensure betas are non-negative.
    off_diag = np.sqrt(betas)  # length should be 2n
    # Compute the eigenvalues of the symmetric tridiagonal matrix.
    roots = eigh_tridiagonal(diag, off_diag, eigvals_only=True)
    return roots

def Hn_eig(X:list[int], fast = False):
    """Gives the eigenvalues of H_n for an arbitrary sequence of X"""
    # Build betas as X[::-1] + X; converting to lists if X is a NumPy array.
    X = np.asarray(X)
    betas = list(X[::-1]) + list(X)
    if fast:
      return compute_poly_roots_tridiag(betas)
    else:
      return compute_poly_roots(betas)

def Fn_eig(X:list[int], fast = False):
    """Gives the eigenvalues of F_n for an arbitrary sequence of X"""
    X = np.asarray(X)
    betas = list(X)
    if fast:
      return compute_poly_roots_tridiag(betas)
    else:
      return compute_poly_roots(betas)

def toeplitz_tridag_eig(beta, dim):
  """Helper function for finding the eigenvalues of p-nary trees"""
  eigs = [2*beta*np.cos(np.pi * i /(dim+1)) for i in range(1, dim+1)]
  return eigs

def pnary_eigs(p,d):
  """Computes the eigenvalues of a p-nary glued tree of depth d"""
  N_ns = [p**i for i in range(0,d+1)]
  # Compute eigenvalues for subproblems.
  Hn_eigs = [0] + [toeplitz_tridag_eig(np.sqrt(p),2*i+1) for i in range(1, d+1)]

  eigs = Hn_eigs[-1]
  for i in range(d):
        mult = int( (p - 1) * N_ns[-1] / N_ns[i+1])
        if(i != 0):
          new_eigs = Hn_eigs[i] * mult       
        else:
          new_eigs = [Hn_eigs[i]] * mult
        eigs = eigs + new_eigs
  return eigs

def pnary_eigs_counter(p,d,tol = 8):
  """p (int): branching factor
     d (int): depth
  """
  #N_ns = [p**i for i in range(0,d+1)]
  Hn_eigs = [[0]] + [np.round(toeplitz_tridag_eig(np.sqrt(p),2*i+1),decimals=tol) for i in range(1, d+1)]

  eig_counter = Counter(Hn_eigs[-1])

  for i in range(d):
    mult = int( (p - 1) * p**(d-i-1))
    for eig in Hn_eigs[i]:
      eig_counter[np.round(eig,decimals=tol)] += mult

  return eig_counter

def A_x_counter(X:list[int],tol=8):
  X = np.asarray(X)
  d = len(X)
  # Compute cumulative products more efficiently.
  N_ns = np.concatenate(([1], np.cumprod(X)))

  # Compute eigenvalues for subproblems.
  Hn_eigs = [np.array([0])] + [np.round(Hn_eig(X[:i],True), decimals= tol)for i in range(1, d+1)]
  # Prepend a zero to X for easier indexing.
  X_with_zero = np.concatenate(([0], X))
  
  # Collect eigenvalue arrays in a list rather than concatenating repeatedly.
  eig_counter = Counter(Hn_eigs[-1])
  for i in range(d):
      mult = int((X_with_zero[i+1] - 1) * int(N_ns[-1] / N_ns[i+1]))
      for eig in Hn_eigs[i]:
        eig_counter[np.round(eig,decimals=tol)] += mult
  
  return eig_counter

def CCAM_counter(X:list[int],tol=4):
  X = np.asarray(X)
  d = len(X)
  # Compute cumulative products more efficiently.
  N_ns = [1,1]+[np.prod(X[1:i]) for i in range(2,len(X)+1)]

  # Compute eigenvalues for subproblems.
  Fn_eigs = [np.array([0])] + [np.round(Fn_eig(X[:i],True), decimals= tol)for i in range(1, d+1)]
  # Prepend a zero to X for easier indexing.
  X_with_zero = np.concatenate(([0], X))
  
  # Collect eigenvalue arrays in a list rather than concatenating repeatedly.
  eig_counter = Counter(Fn_eigs[-1])
  eig_counter.update(eig_counter)

  for i in range(2,d+1):
      mult = 2* int( (X_with_zero[i] - 1) * int(N_ns[-1] / N_ns[i]))
      for eig in Fn_eigs[i-1]:
          eig_counter[np.round(eig,decimals=tol)] += mult

  zero_dim = (X_with_zero[1]-2) * N_ns[d]
  if zero_dim > 0:
      eig_counter[0] += zero_dim
  
  return eig_counter

def pnary_CCAM_counter(p, d,tol=4):
  # Compute eigenvalues for subproblems.
  #[np.round(toeplitz_tridag_eig(np.sqrt(p),i+1),decimals=tol) for i in range(1, d+1)]
  Fn_eigs = [np.array([0])] + [np.round(toeplitz_tridag_eig(np.sqrt(p),i+1),decimals=tol) for i in range(1, d+1)]
  
  # Collect eigenvalue arrays in a list rather than concatenating repeatedly.
  eig_counter = Counter(Fn_eigs[-1])
  eig_counter.update(eig_counter)

  for i in range(2,d+1):
      mult = 2*int( (p - 1) * p**(d - i) )
      for eig in Fn_eigs[i-1]:
          eig_counter[np.round(eig,decimals=tol)] += mult

  zero_dim = (p-2) * p**(d-1)
  if zero_dim > 0:
      eig_counter[0] += zero_dim
  
  return eig_counter


def format_counter(counter):

    """Formats A Counter Objects for Easy Step Plotting"""
    # Example: a counter whose keys are numeric values (e.g. eigenvalues)
    total = sum(counter.values())

    # Sort keys in ascending order (or in any desired order)
    keys = sorted(counter.keys())

    # Compute the fraction of the total for each key
    fractions = [counter[k] / total for k in keys]

    # Compute cumulative boundaries along x: start at 0 and add each fraction
    cum = np.concatenate(([0], np.cumsum(fractions)))

    x_steps = []
    y_steps = []
    for i, k in enumerate(keys):
        x_steps.extend([cum[i], cum[i+1]])
        y_steps.extend([k, k])

    return (x_steps,y_steps)
