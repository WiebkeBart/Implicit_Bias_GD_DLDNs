""""This module provides the Gradient Descent function"""
import numpy as np

def GD(x0, loss, grad_loss, eta, eps=10e-4, T=int(10e7), output=False):
    """Vanilla Gradient Descent update scheme: x_{t+1}=x_t - eta_t \nabla L(x_t)

    Parameters:
        x0 (np.array size N)            - initial value to start the algorithm
        loss (fct: R^N \to R_+)         - loss function, gets (only!) x_t provides L(x_t)
        grad_loss (fct: R^N \to R^N)    - gradient of the loss function, gets (only!) x_t provides \nabla L(x_t)
        eta (fct: R^N \to R_+)          - learning rate fct, for eta_t adaptive (depends only on x_t), constant for constant
        eps (default 10e-4)             - threshold for stopping L(x_t) \leq eps
        T  (default int(10e7))          - maximal number of iterations
        output (boolean)                - whether the algorithm output should be printed or not
    Returns:
        x_values                        - list of x_t
        loss_values                     - list of L(x_t)
        grad_norm                       - list of ||\nabla L(x_t)||
        grad_values                     - list of L(x_t)
        eta_values                      - list of eta_t (only interesting if eta is a fct, otherwise constant)

    """

    # preparing empty lists to save the corresponding values later
    x_values = []
    loss_values = []
    grad_norm = []
    grad_values = []
    eta_values = []

    # first entries for the list
    x_values.append(x0)
    loss_values.append(loss(x_values[-1]))
    gradient = grad_loss(x_values[-1])
    grad_norm.append(np.linalg.norm(gradient))
    grad_values.append(gradient)
    eta_values.append(eta(x_values[-1]))

    for t in range(T):

        # Calculate the new iterate x_t and the corresponding loss (L(x_t)
        x_values.append(x_values[-1] - eta_values[-1] * gradient)
        loss_values.append(loss(x_values[-1]))

        # if output==True we provide some information along the training procedure
        if output:
            print("Iteration: ", t)
            print("\t \t loss_values =", loss_values[-1])
            print("\t \t grad_norm =", grad_norm[-1])
            print("\t \t eta_value =", eta_values[-1])

        # append gradient and gradient norm to the list
        gradient = grad_loss(x_values[-1])
        grad_norm.append(np.linalg.norm(gradient))
        grad_values.append(gradient)
        eta_values.append(eta(x_values[-1]))

        # Stopping criterion
        if loss_values[-1] <= eps:#or np.linalg.norm(grad_values[-1]) <= 0.01*eps:
            print('Convergence criterion met.')
            return x_values, loss_values, grad_norm, grad_values, eta_values
    print('No convergence.')
    return x_values, loss_values, grad_norm, grad_values, eta_values

def SGD(x0, full_loss, batch_loss, grad_loss_SGD, batch_size, M, eta, eps=10e-4,  T=int(10e7), output=False):
    """SGD update scheme: x_{t+1} = x_t - eta_t \nabla L_{B_t}(x_t)

    Parameters:
        x0 (np.array size N)                            - initial point to start the algorithm
        full_loss  (fct: R^N \to R)                     - full loss function L(x_t) parameter x
        batch_loss (fct: R^N \times B \to R             - bach loss L_{B_t}(x_t) parameter x and indices
        grad_loss_SGD (fct: R^N \times B \to R          - gradient of batch loss
        M                                               - integer, maximal batch-size
        eta  (fct: R^N \to R_+                          - learning rate fct, for eta_t adaptive (depends only on x_t), constant for constants
        eps (default 10e-4)                             - threshold for stopping L(x_t) \leq eps
        T  (default int(10e7))                          - maximal number of iterations
        output (boolean)                                - whether the algorithm output should be printed or not
    """

    # preparing empty lists to save the corresponding values later
    x_values = []
    loss_full_values = []
    loss_sgd_values = []
    grad_norm = []
    grad_values = []
    batches = [] #save each batch for calculations later
    eta_values = []

    # Get the batch for sgd
    indices = np.random.permutation(M)
    indices = indices[:batch_size]
    indices.sort()
    batches.append(indices)

    print(indices)

    x_values.append(x0)
    loss_full_values.append(full_loss(x_values[-1]))
    loss_sgd_values.append(batch_loss(x_values[-1], indices))

    gradient = grad_loss_SGD(x_values[-1], indices)

    grad_norm.append(np.linalg.norm(gradient))
    grad_values.append(gradient)
    eta_values.append(eta(x_values[-1]))

    for t in range(T):
        # Calculating the new entry and the corresponding loss
        x_values.append(x_values[-1] - eta_values[-1] * gradient)


        #Draw a new batch
        indices = np.random.permutation(M)
        indices = indices[:batch_size]
        indices.sort()
        batches.append(indices)
        #print(indices)

        loss_full_values.append(full_loss(x_values[-1]))
        loss_sgd_values.append(batch_loss(x_values[-1], indices))

        # if output==True we print some information along the training
        if output:
            print("Iteration: ", t)
            print("\t \t full loss =", loss_full_values[-1])
            print("\t \t sgd loss =", loss_sgd_values[-1])
            print("\t \t grad_norm =", grad_norm[-1])

        gradient = grad_loss_SGD(x_values[-1], indices)
        grad_norm.append(np.linalg.norm(gradient))
        grad_values.append(gradient)
        eta_values.append(eta(x_values[-1]))

        if loss_full_values[-1] <= eps:
            print('Convergence criterion met.')
            return x_values, loss_full_values, loss_sgd_values, grad_norm, grad_values, eta_values, batches
    print('No convergence.')
    return x_values, loss_full_values, loss_sgd_values, grad_norm, grad_values, eta_values, batches
