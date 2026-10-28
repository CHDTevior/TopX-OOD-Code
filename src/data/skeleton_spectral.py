"""Heat-kernel signature of a skeleton tree: the joint coordinate of HK-RoPE.

For a tree with J joints, L = I - D^{-1/2} A D^{-1/2} is the symmetric normalised Laplacian of its adjacency A. With
the eigenpairs (lambda_k, v_k), 0 = lambda_0 < lambda_1 <= ... <= lambda_{J-1}, the signature of joint j at diffusion
time tau is

    h_j(tau) = sum_{k >= 1} exp(-tau lambda_k) v_kj^2,

the diagonal of the heat kernel exp(-tau L) without its zero-eigenvalue mode. Summed over an eigenspace, v_kj^2 is a
diagonal entry of the eigenspace's projector, so h does not depend on the eigenvector signs or on the basis chosen
inside a repeated eigenvalue, and a relabelling of the joints only permutes its rows. Every mode enters, so the width
is the number of diffusion times whatever J is. Each column is divided by its mean over the joints of the rig.
"""
import numpy as np


def hks_scales(num_scales: int = 8) -> np.ndarray:
    """Diffusion times tau_i = 2^(i-2), i = 0 .. num_scales-1 (0.25, 0.5, 1, ..., 32 for eight scales)."""
    return 0.25 * (2.0 ** np.arange(int(num_scales), dtype=np.float64))


def tree_laplacian(parents) -> np.ndarray:
    """parents [J] (root: -1 or itself) -> the symmetric normalised Laplacian [J, J] float64 of the tree."""
    parents = np.asarray(parents, dtype=np.int64).ravel()
    n = int(parents.shape[0])
    adjacency = np.zeros((n, n), dtype=np.float64)
    for child, parent in enumerate(parents.tolist()):
        if parent >= 0 and parent != child:
            adjacency[child, parent] = adjacency[parent, child] = 1.0
    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = np.where(degree > 0, 1.0 / np.sqrt(np.where(degree > 0, degree, 1.0)), 0.0)
    return np.eye(n) - inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]


def heat_kernel_signature_raw(parents, scales) -> np.ndarray:
    """[J, S] float64: h_j(tau) for every joint j and diffusion time tau in `scales`."""
    eigenvalues, eigenvectors = np.linalg.eigh(tree_laplacian(parents))
    lam = np.maximum(eigenvalues[1:], 0.0)                   # the J-1 non-trivial eigenvalues
    weights = np.real(eigenvectors[:, 1:]) ** 2              # [J, J-1]
    tau = np.asarray(scales, dtype=np.float64).ravel()
    return weights @ np.exp(-np.outer(lam, tau))


def heat_kernel_signature(parents, scales) -> np.ndarray:
    """[J, S] float32: the signature with every column divided by its mean over the joints (zeros for one joint)."""
    h = heat_kernel_signature_raw(parents, scales)
    column_mean = h.mean(axis=0, keepdims=True)
    return np.divide(h, column_mean, out=np.zeros_like(h), where=column_mean > 0).astype(np.float32)
