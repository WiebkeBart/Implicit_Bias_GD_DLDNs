""""module provides useful functions for Linear Diagonal Networks (LDNs)

    Functions:
    Loss                   --  loss function for end-to-end parameter beta
    Loss_L                 --  loss function for u (positive model)
    Loss_uv                --  loss function for u, v (difference model)
    gradient               --  gradient of loss function for end-to-end parameter beta
    gradient_L             --  gradient of loss function Loss_L (positive model)
    gradient_uv            --  gradient of loss function Loss_uv (difference model)
"""

import numpy as np

def Loss(beta, ATA, yTA, yTy, M):
    """Calculates the loss for the end-to-end parameter beta
        Formula 1/(2M)||A\beta - y||_2^2 = 1/(2M)(\beta^T A^TA \beta + y^Ty) - 1/M y^TA\beta

    Parameters:
        beta      -- numpy vector of size N, point at which to calculate the loss
        ATA       -- numpy array of size N\times N, A^TA
        yTA       -- numpy vector of size N, y^TA
        yTy       -- scalar, y^Ty
        M         -- integer, size of y
    """

    return 1/(2*M)*(np.matmul(np.transpose(beta), np.matmul(ATA, beta)) - 2*np.matmul(yTA, beta) + yTy)

def Loss_L(x, ATA, yTA, yTy, L, M):
    """Calculates the loss for the inner parameter x (positive model)
           Formula 1/(2M)||Ax^L - y||_2^2 = 1/(2M)(\beta^T A^TA \beta + y^Ty) - 1/My^TA\beta
           Use the fct for the end-to-end parameter beta

       Parameters:
           x       -- numpy vector of size N, point at which to calculate the loss
           ATA     -- numpy array of size N\times N, A^TA
           yTA     -- numpy vector of size N, y^TA
           yTy     -- scalar, y^Ty
           L       -- scalar, depth of the network (integer)
           M       - integer, size of y
           """

    beta = x**L
    return Loss(beta, ATA, yTA, yTy, M)

def K_L(x, ATA, yTA, yTy,M):
    " x is of shape (L,N)"
    beta = np.product(x, axis=0)
    return Loss(beta, ATA, yTA, yTy, M)

def batch_loss_L(x, indices, A, y, L):
    """For a given batch (indices) calculates the corresponding batch loss, by restricting A, y appropriately and linking batch to the loss fct"""
    #print(indices)
    beta = x**L
    A_restricted = A[indices, :]
    y_restricted = y[indices]
    M = len(indices)
    #print(M)
    return Loss(beta, A_restricted.T@A_restricted, y_restricted.T@A_restricted, y_restricted.T@y_restricted, M)

def batch_loss_uv(x, indices, A, y, L):
    """For a given batch (indices) calculates the corresponding batch loss, by restricting A, y appropriately and linking batch to the loss fct"""

    u = x[0]
    v = x[1]
    M = len(indices)
    beta = u**L - v**L
    A_restricted = A[indices, :]
    y_restricted = y[indices]
    return Loss(beta, A_restricted.T @ A_restricted, y_restricted.T @ A_restricted, y_restricted.T @ y_restricted, M)

# Loss function for the inner layers; unbiased case
def Loss_uv(x, ATA, yTA, yTy, L, M):
    """Calculates the loss for the inner parameter x (difference model) beta= u^L-v^L
           Formula 1/(2M)||A(x[0]^L-x[1]^L) - y||_2^2 = (1/2M)(\beta^T A^TA \beta + y^Ty - 2y^TA\beta)
           Use the fct for the end-to-end parameter beta

       Parameters:
           x       -- numpy vector of size (2,N), point at which to calculate the loss u=x[0], v=x[1]
           ATA     -- numpy array of size N\times N, A^TA
           yTA     -- numpy vector of size N, y^TA
           yTy     -- scalar, y^Ty
           L       -- scalar, depth of the network
           """

    u = x[0]
    v = x[1]
    beta = u**L - v**L
    return Loss(beta, ATA, yTA, yTy, M)



# Gradient w.r.t. the outer product
def gradient(beta, ATA, ATy, M):
    """Calculates the gradient of the loss for the end-to-end parameter beta
            Formula 1/M A^T(A\beta -y)= 1/M (A^TA\beta - A^Ty)
    """

    return 1/M*(np.matmul(ATA, beta)-ATy)


# Gradient w.r.t. the inner layer; biased case
def gradient_L(x, ATA, ATy, L, M):
    """Calculates the gradient of the loss for the inner parameter x (postive model)
     uses the gradient for beta in the calculation
    Formula L/M A^T(Ax^L -y)(x^(L-1))  = L \nabla L(\beta)\odot x^(L-1)
    """

    beta = x**L
    grad =  gradient(beta, ATA, ATy, M)
    return L*grad*(x**(L-1))

def gradient_K(x, ATA, ATy, M):
    'x has shape (L,N)'
    beta = np.product(x, axis=0)
    grad = gradient(beta, ATA, ATy, M)
    left = np.ones_like(x)
    right = np.ones_like(x)
    left[1:] = np.cumprod(x[:-1], axis=0)
    right[:-1] = np.cumprod(x[1:][::-1], axis=0)[::-1]
    return (left*right)*grad

def gradient_SGD(beta, indices, A, y):
    """Calculates the gradient of the loss for the outer parameter beta for a certain batch (indices)
    """

    A_restricted = A[indices, :]
    y_restricted = y[indices]
    k = len(indices)
    return gradient(beta, A_restricted.T @ A_restricted, A_restricted.T @ y_restricted, k)


def gradient_SGD_L(x, indices, A, y, L):
    """For a given batch (indices) calculates the corresponding batch gradient, by restricting A, y appropriately and linking batch to the gradient fct"""

    A_restricted = A[indices, :]
    y_restricted = y[indices]
    k = len(indices)
    grad = gradient_SGD(x**L, indices, A, y)
    return L*grad*(x**(L-1))


def gradient_u_v(x, ATA, ATy, L, M):
    """Calculates the gradient of the loss for the inner parameter x (difference model)
     uses the gradient for beta in the calculation
    Formula 1/M LA^T(Ax^L -y)(x^(L-1))  = L \nabla L(\beta)\odot x^(L-1)
    """

    u = x[0]
    v = x[1]
    beta = u**L - v**L
    grad = gradient(beta, ATA, ATy, M)
    grad_u = L*grad*(u**(L-1))
    grad_v = -L*grad*(v**(L-1))
    return np.array((grad_u, grad_v))

def gradient_uv_SGD(x, indices, A, y, L):
    u = x[0]
    v = x[1]
    beta = u ** L - v ** L
    A_restricted = A[indices, :]
    y_restricted = y[indices]
    M = len(indices)
    grad = gradient(beta, A_restricted.T@A_restricted, A_restricted.T@y_restricted, M)
    grad_u = L * grad * (u ** (L - 1))
    grad_v = -L * grad * (v ** (L - 1))
    return np.array((grad_u, grad_v))