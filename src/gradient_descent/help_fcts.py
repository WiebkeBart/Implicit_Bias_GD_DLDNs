import numpy as np

def Bregman_distance(beta_star, beta, L):
    """function to calculate the Bregman distance, R_+^N X R_+^N x N ----> R_{>=0}
        For the unbalanced case, one should use the sum of the two bregman divergences, split between beta_+, beta_- and u^L, v^L.

    Parameters:
        beta_star       -   np.array of size N (solution to the regression problem)
        beta            -   np.array of size N
        L               -   integer scalar
    """

    if L>2:
        factor = 1./(2-L)
        factor2 = 2/L
        return factor*(0.5*np.inner(np.power(beta_star, factor2), np.ones_like(beta_star))- 1./L *np.inner(beta_star, np.power(beta,factor2-1)))+ 1./(2*L)*np.inner(np.power(beta, factor2), np.ones_like(beta))
    if L==2:
        print(beta_star, beta)
        return 0.25*(np.inner(beta_star, np.log((beta_star+0.000001*np.ones_like(beta_star))/beta)-np.ones_like(beta_star))+np.inner(beta, np.ones_like(beta)))

def H(eta, beta, grad, L):
    """ calculating the error term per iterate.

    Parameters:
        eta             -   scalar, positive float
        beta            -   np.array of size N
        grad            -   np.array of size N, gradient of beta
        L               -   integer scalar
    """

    if L>2:
        factor =  1-2/L
        beta_scaled = beta**factor
        beta_scaled_invers = beta**(2/L-1)
        return 1./(L*(L-2))*(1-(1-eta*L*grad*beta_scaled)**(2-L))*beta_scaled_invers + eta*grad

    if L==2:
        return 0.5*np.log(1-2*eta*grad) + eta*grad

def assumption1(L, grad_norm, beta_inf):
    """Function gives the upper bound in assumption 1 in the paper. 
        N x R_+ x R_+  ----> R_+

    Parameters:
       L            -   integer, depth of the network
       grad_norm    -   scalar, \ell_2 norm of the gradient at the beta
       beta_inf     -   scalar, \ell_\inf norm of the current beta
    Returns:
        float scalar >0
    """
    
    factor = 1-2/L
    denominator = 2*L*L*grad_norm*np.power(beta_inf, factor)
    return 1/denominator

def assumption2(L, grad_norm, beta_inf, beta_2, sol_2, loss, theta=1):
    """Function gives the upper bound in assumption 2 in the paper. (for theta=1)
            N X R_+ X R_+ X R_+ X R_+ X R_+  ----> R_+

        Parameters:
           L            -   integer, depth of the network
           grad_norm    -   scalar, \ell_2 norm of the gradient at the beta
           beta_inf     -   scalar, \ell_\inf norm of the current beta
           beta_2       -   scalar, \ell_2 norm of the current beta
           sol_2        -   scalar, \ell_2 norm of a solution
           loss         -   current loss evaluated at beta
        Returns:
            float scalar >0
    """

    factor = 1-2/L
    sqrt_e = np.sqrt(np.exp(1))
    pre_factor = 16/9*(L-1)
    denominator = np.power(beta_inf,factor)*grad_norm*grad_norm*L*(sqrt_e*beta_inf*L+pre_factor*(sqrt_e*beta_2+sol_2))
    return (theta*loss)/denominator


def adaptive_step_size(x, L, sol_2, grad_fct, loss_fct, theta=1):
    """Calculates the adaptive step-size for the positive case

    Parameters:
        x               -   np.array of size N, current iterate (is x)
        L               -   integer, depth of the network
        sol_2           -   float, ell_2 norm of a solution
        grad_fct        -   fct of the gradient function, input of that function is np.array of size N,
                            returns np.array of size N, should be the gradient of beta! not x!
        loss_fct        -   fct of the loss fct, input of that function is np.array of size N
                            returns non-negative float
    Returns:
        non-negative float
    """

    beta = x**L
    gradient = grad_fct(beta)
    grad_norm = np.linalg.norm(gradient, ord=2)
    beta_inf = np.linalg.norm(beta, ord=np.inf)
    beta_2 = np.linalg.norm(beta, ord=2)
    loss = loss_fct(beta)

    eta_1 = assumption1(L, grad_norm, beta_inf)
    eta_2 = assumption2(L, grad_norm, beta_inf, beta_2, sol_2, loss, theta=theta)
    return min(eta_1, eta_2)

def assumption2_1(L, grad_norm, u_inf, v_inf):
    """ Calculates the first step-size bound for the general case

    Parameters:
        L            -      integer, depth of the network
        grad_norm    -      float, \ell_2 norm of the gradient at the beta
        u_inf        -      float, \ell_\inf norm of the current u-value
        v_inf        -      float, \ell_\inf norm of the current v-value

    """
    max_norm = max(u_inf, v_inf)
    denominator = 2*L*L*grad_norm*np.power(max_norm, L-2)
    return 1/denominator

# the same as assumption2 just for the (u,v) case
def assumption2_2(L, grad_norm, u_inf, v_inf, u_2, v_2, sol_plus_2, sol_minus_2, loss):
    """Function gives the upper bound in assumption 2 in the paper. for theta = 1/2

    Parameters:
        L               -   integer, depth of the network
        grad_norm       -   scalar, \ell_2 norm of the gradient at the beta=u**L - v**L
        u_inf           -   scalar, \ell_2 norm of the current u
        v_inf           -   scalar, \ell_2 norm of the current v
        u_2             -   scalar, \ell_2 norm of the current u**L
        v_2             -   scalar, \ell_2 norm of the current v**L
        sol_plus_2      -   scalar, \ell_2 norm of the positive part of a solution
        sol_minus_2     -   scalar, \ell_2 norm of the negative part of a solution
        loss            -   scalar, current loss evaluated at beta
    Returns:
        float scalar non-negative
    """

    sqrt_e = np.sqrt(np.exp(1))
    def which_bound(a_inf, a_2, b): #small fct calculating the two options for u,v
        return a_inf**(L-2)*(sqrt_e*L*a_inf**L+(L-1)*16/9*(sqrt_e*a_2+b))
    alpha = max(which_bound(u_inf, u_2,sol_plus_2), which_bound(v_inf, v_2, sol_minus_2))
    return loss/(4*L*L*grad_norm*grad_norm*alpha)

def adaptive_step_size_difference(u, v, L, sol_plus_2, sol_minus_2, grad_fct, loss_fct):
    """Calculates the adaptive step-size for the difference case

    Parameters:
        u               -   np.array of size N, current iterate (is u)
        v               -   np.array of size N, current iterate (is v)
        sol_plus_2      -   float, ell_2 norm of a positive part of a solution
        sol_minus_2     -   float, ell_2 norm of a negative part of a solution
        grad_fct        -   fct of the gradient function, input of that function is np.array of size N,
    """
    beta = u**L -v**L
    gradient = grad_fct(beta)
    grad_norm = np.linalg.norm(gradient, ord=2)
    u_inf = np.linalg.norm(u, ord=np.inf)
    v_inf = np.linalg.norm(v, ord=np.inf)
    u_2 = np.linalg.norm(u**L, ord=2)
    v_2 = np.linalg.norm(v**L, ord=2)
    eta_1 = assumption2_1(L, grad_norm, u_inf, v_inf)
    eta_2 = assumption2_2(L, grad_norm, u_inf, v_inf, u_2, v_2, sol_plus_2, sol_minus_2, loss_fct(beta))
    return min(eta_1, eta_2)

def perturbed_init(beta_0, L, delta):
    """Calculates beta_0 tilde given beta_0, L and delta

    Parameters:
        beta_0       -   np.array of size N
        L            -   integer, depth of the network
        delta        -  np.array of size N

        returns beta_0_tilde        np.array of size N
        """
    if L==2:
        return beta_0 * np.exp(delta)
    else:
        factor = 2/L-1
        return np.power(np.power(beta_0, factor) + delta, L/(2-L))

def perturbed_init_uv(u, delta, L):
    """Calculates u_0 or v_0 tilde given u_0, L and delta

       Parameters:
           u_0       -   np.array of size N
           L            -   integer, depth of the network
           delta        -  np.array of size N

           returns u_0_tilde        np.array of size N
           """
    if L==2:
        return (u**2*np.exp(delta))**(1/2)
    else:
        return np.power(u**(2-L)+delta, 1/(2-L))

def delta(H_sum: float,  L: int) -> float:
    """ Given the sum of error terms H_sum and the depth of the network L,
    this returns the error delta"""

    if L==2:
        return -4*H_sum
    if L>2:
        return -L*(L-2)*H_sum


def epsilon(beta_0_tilde, Q, L):
    gamma = 1 - 2/L
    scaled = beta_0_tilde**(2/L)
    numerator = np.linalg.norm(scaled, ord=1)**gamma - np.min(scaled)**gamma
    denominator = Q**gamma - np.linalg.norm(scaled, ord=1)**gamma
    return L/2*numerator/denominator
