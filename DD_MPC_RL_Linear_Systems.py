import numpy as np
import control
import cvxpy as cp
import scipy.io
from MDL_sim_prestab import MDL_sim_prestab
from henkel_r import henkel_r
import matplotlib.pyplot as plt

def generate_initial_io(sys, ui, xi0, n, M):
    for i in range(n-1):
        xi[:, i+1] = sys.A @ xi[:, i] + sys.B @ ui[:, i]

    yi = sys.C @ xi + sys.D @ ui
    K = np.eye(m) # pre-stabilizing controller
    W_init = np.zeros((M, 1))
    return xi, yi, K, W_init

def define_terminal_constraints(sys, m, n):
    # Terminal constraints
    u_term = np.ones((m, 1))
    y_term = sys.C @ np.linalg.inv(np.eye(n) - sys.A) @ sys.B @ u_term

    return u_term, y_term

def define_cost_matrices(m, p):

    R = 1e-4 * np.eye(m)  # input weighting
    Q = 3 * np.eye(p)  # output weighting
    S = np.zeros((m, p))

    return R, Q, S

def construct_pi_matrix(Q, R, S, L, m, p):
    # Constructing the Pi matrix
    Pi = np.block([
        [np.kron(np.eye(L), R), np.kron(np.eye(L), S)],
        [np.kron(np.eye(L), S.T), np.kron(np.eye(L), Q)]
    ])
    return Pi

def construct_hankel_matrices(u, y, L, N, m, p):

    Hu = henkel_r(u.flatten(), L, N-L+1, m)
    Hy = henkel_r(y.flatten(), L, N-L+1, p)

    return Hu, Hy

def Phi_ybar(ybar):

    y1_k = ybar[0]
    y2_k = ybar[1]
    value_Phi = np.array([
        y1_k, y2_k, y1_k**2, y2_k**2, y1_k*y2_k, y1_k**3, y2_k**3, y1_k**2 * y2_k, y1_k * y2_k**2
    ])
    return value_Phi

def J_Phi_ybar(ybar):

    J = np.zeros((9, 2))
    # Each row of J is the gradient of the corresponding element of Phi
    J[0, :] = [1, 0]               # d(y1)/d(y1), d(y1)/d(y2)
    J[1, :] = [0, 1]               # d(y2)/d(y1), d(y2)/d(y2)
    J[2, :] = [2*y[0], 0]          # d(y1^2)/d(y1), d(y1^2)/d(y2)
    J[3, :] = [0, 2*y[1]]          # d(y2^2)/d(y1), d(y2^2)/d(y2)
    J[4, :] = [y[1], y[0]]         # d(y1*y2)/d(y1), d(y1*y2)/d(y2)
    J[5, :] = [3*y[0]**2, 0]       # d(y1^3)/d(y1), d(y1^3)/d(y2)
    J[6, :] = [0, 3*y[1]**2]       # d(y2^3)/d(y1), d(y2^3)/d(y2)
    J[7, :] = [2*y[0]*y[1], y[0]**2] # d(y1^2*y2)/d(y1), d(y1^2*y2)/d(y2)
    J[8, :] = [y[1]**2, 2*y[0]*y[1]] # d(y1*y2^2)/d(y1), d(y1*y2^2)/d(y2)

    return J

def VFA_cost_term(W, ybar, x, N, L, m, p):
    y_bar2 = ybar[:2]
    Phi_ybar_val = Phi_ybar(y_bar2)
    J_Phi_ybar_val = J_Phi_ybar(y_bar2)
    # Extract y from the decision variable x
    y = x[N-L+1+m*L:N-L+1+(m+p)*L]  
    y2 = y[:2]
    # Compute the cost term
    cost_term = W.T @ Phi_ybar_val + W.T @ J_Phi_ybar_val @ (y2 - y_bar2)
    return cost_term


def initialize_open_loop_variables(T, L, N, m, p, robust):
    u_ol = np.zeros((m * L, T))
    y_ol = np.zeros((p * L, T))
    sigma_ol = np.zeros((p * L, T))
    alpha_ol = np.zeros((N - L + 1, T))
    u_init_store = np.zeros((m * nu, T))
    y_init_store = np.zeros((p * nu, T))
    fval = np.zeros(T)
    if robust:
        sol_store = np.zeros(((m + 2 * p) * L + N - L + 1, T))

    return u_ol, y_ol, sigma_ol, alpha_ol, u_init_store, y_init_store, sol_store, fval

# Constants and Parameters
robust = True
TEC = True
tol_opt = 1e-4
opt_settings = {'OptimalityTolerance': 1e-9, 'MaxIterations': 20000, 'ConstraintTolerance': 1e-9}

# System Specifications
n, nu, M, noise_max, N, L_true, T, m, p, alpha, epsilon_W = 4, 4, 9, 0.02, 400, 30, 1000, 2, 2, 0.0001, 0.0000001
L = L_true + nu

# Load System Data
data = scipy.io.loadmat('tank_sys2.mat')
A_d, B_d, C, D, T_s = [data[key] for key in ['A_d', 'B_d', 'C', 'D', 'T_s']]
T_s = T_s.item()  # Assuming T_s is a scalar
sys = control.ss(A_d, B_d, C, D, T_s)

u_term, y_term = define_terminal_constraints(sys, m, n)
R, Q, S = define_cost_matrices(m, p)
Pi = construct_pi_matrix(Q, R, S, L, m, p)

# Cost for QP
if robust:
    lambda_sigma = 1e3
    lambda_alpha = 1e-1
    H = 2 * np.block([
        [lambda_alpha * np.eye(N-L+1), np.zeros((N-L+1, (m+p)*L)), np.zeros((N-L+1, p*L))],
        [np.zeros(((m+p)*L, N-L+1)), Pi, np.zeros(((m+p)*L, p*L))],
        [np.zeros((p*L, N-L+1 + (m+p)*L)), lambda_sigma * np.eye(p*L)]
    ])
    f = np.concatenate([
        np.zeros((N-L+1, 1)),
        -2 * np.kron(np.eye(L), R) @ np.tile(u_term, (L, 1)),
        -2 * np.kron(np.eye(L), Q) @ np.tile(y_term, (L, 1)),
        np.zeros((p*L, 1))
    ])
    
# Initial I/O Trajectories
ui, xi0 = np.random.rand(m, n), np.random.rand(n, 1)
xi = np.zeros((n, n))
xi[:, 0] = xi0.squeeze()
xi, yi, K, W_init = generate_initial_io(sys, ui, xi0, n, M)

# MPC Simulation
u, x, y = MDL_sim_prestab(sys, ui, yi, K, noise_max, 0, N)

# Constructing the Hankel Matrices
Hu, Hy = construct_hankel_matrices(u, y, L, N, m, p)

u_max = np.inf * np.ones((m, 1))
u_min = -np.inf * np.ones((m, 1))
y_max = np.inf * np.ones((p, 1))
y_min = -np.inf * np.ones((p, 1))

if robust:
    sigma_max = np.inf * np.ones((p, 1))
    sigma_min = -sigma_max
    ub = np.concatenate([
        np.full((N-L+1, 1), np.inf),
        np.tile(u_max, (L, 1)),
        np.tile(y_max, (L, 1)),
        np.tile(sigma_max, (L, 1))
    ])
    lb = np.concatenate([
        np.full((N-L+1, 1), -np.inf),
        np.tile(u_min, (L, 1)),
        np.tile(y_min, (L, 1)),
        np.tile(sigma_min, (L, 1))
    ])

  # With terminal constraints
    B = np.block([
        [Hu, -np.eye(m*L), np.zeros((p*L, p*L)), np.zeros((p*L, p*L))],
        [Hy, np.zeros((p*L, m*L)), -np.eye(p*L), -np.eye(p*L)],
        [np.zeros((m*nu, N-L+1)), np.hstack([np.eye(m*nu), np.zeros((m*nu, m*(L-nu)))]), np.zeros((m*nu, p*L)), np.zeros((m*nu, p*L))],
        [np.zeros((p*nu, N-L+1)), np.zeros((p*nu, m*L)), np.hstack([np.eye(p*nu), np.zeros((p*nu, p*(L-nu)))]), np.hstack([np.zeros((p*nu, p*nu)), np.zeros((p*nu, p*(L-nu)))])],
        [np.zeros((m*nu, N-L+1)), np.hstack([np.zeros((m*nu, m*(L-nu))), np.eye(m*nu)]), np.zeros((m*nu, p*L)), np.zeros((m*nu, p*L))],
        [np.zeros((p*nu, N-L+1)), np.zeros((p*nu, m*L)), np.hstack([np.zeros((p*nu, p*(L-nu))), np.eye(p*nu)]), np.zeros((p*nu, p*L))]
    ])

# MPC Loop
u_init = 0.5 * np.ones((m, nu))  # initial input
x0 = np.array([0.4, 0.4, 0.5, 0.5])  # initial state
x_init = np.zeros((n, nu))
x_init[:, 0] = x0
for i in range(nu-1):
    x_init[:, i+1] = sys.A @ x_init[:, i] + sys.B @ u_init[:, i]

y_init = sys.C @ x_init + sys.D @ u_init
u_cl = np.zeros((m, T))
u_cl[:, :nu] = u_init  # Set initial input

y_cl = np.zeros((p, T))
y_cl[:, :nu] = y_init  # Set initial output

y_cl_noise = np.copy(y_cl)  # Copy of y_cl for noisy output

x_cl = np.zeros((n, T))
x_cl[:, 0] = x0  # Set initial state

# Simulate first nu steps
for j in range(nu):
    x_cl[:, j+1] = sys.A @ x_cl[:, j] + sys.B @ u_cl[:, j]


u_ol, y_ol, sigma_ol, alpha_ol, u_init_store, y_init_store, sol_store, fval = initialize_open_loop_variables(T, L, N, m, p, robust)
# Candidate solution storage variables
u_cand = np.copy(u_ol)
y_cand = np.copy(y_ol)
alpha_cand = np.copy(alpha_ol)
fval_cand = np.zeros((1, T))

# --- Function Definitions Below ---

W = W_init
ybar = np.zeros((68,1))
# Fill the matrix with 0.4 for each element
ybar = np.full_like(ybar, 0.7)
W_history = []
converged = False
# MPC Loop
for j in range(nu + 1, T, M):
    print(j)
      # To display the current iteration
    # Update equality constraints
    c = np.concatenate([
        np.zeros(((m + p) * L, 1)), 
        u_init.flatten()[:, np.newaxis],  # Convert to 2D column vector
        y_init.flatten()[:, np.newaxis],  # Convert to 2D column vector
        np.tile(u_term, (nu, 1)), 
        np.tile(y_term, (nu, 1))
                        ], axis=0)

    # Define and solve the QP problem using cvxpy or another QP solver
    x = cp.Variable((H.shape[0], 1))
    sol = np.zeros(x.shape)
    cost_W = VFA_cost_term(W, ybar, x, N, L, m, p)
    y_current = x[N-L+1+m*L:N-L+1+(m+p)*L]

    # Calculate Phi and J_Phi for ybar
    Phi_ybar_val = Phi_ybar(ybar)
    J_Phi_ybar_val = J_Phi_ybar(ybar)

    cost_W = W.T @ (Phi_ybar_val + J_Phi_ybar_val @ (y_current[:2] - ybar[:2]))

    # Define the objective function including the VFA cost term
    objective = cp.Minimize(0.5 * cp.quad_form(x, H) + f.T @ x + cost_W)
    constraints = [B @ x == c, lb <= x, x <= ub] 

    # Solver options for OSQP

    prob = cp.Problem(objective, constraints)
    osqp_options = {
        'eps_abs': 1e-5,  # Absolute tolerance
        'eps_rel': 1e-5,  # Relative tolerance
        'max_iter': 50000  # Maximum iterations
    }

    # Solve the problem with adjusted settings
    result = prob.solve(solver=cp.OSQP, **osqp_options)

    if prob.status in ["infeasible", "unbounded"]:
        raise ValueError("Optimization problem not solved exactly..")
    if prob.status in ["optimal", "optimal_inaccurate"]:

        sol = x.value.flatten()
        #fval[j] = prob.value + np.dot(np.tile(y_term, (L, 1)).T, np.kron(np.eye(L), Q) @ np.tile(y_term, (L, 1))) + np.dot(np.tile(u_term, (L, 1)).T, np.kron(np.eye(L), R) @ np.tile(u_term, (L, 1)))
        sol_store[:, j] = sol
        alpha_ol[:, j] = sol[:N-L+1]
        u_ol[:, j] = sol[N-L+1:N-L+1+m*L]
        y_ol[:, j] = sol[N-L+1+m*L:N-L+1+(m+p)*L]
        if robust:
            sigma_ol[:, j] = sol[N-L+1+(m+p)*L:N-L+1+(m+2*p)*L]

        u_init_store[:, j-nu] = u_init.flatten()
        y_init_store[:, j-nu] = y_init.flatten()
    # Simulate closed loop
    for k in range(j, min(j + M, T - 1)):
        u_cl[:, k] = u_ol[m*n + (k - j) * m : m*n + m + (k - j) * m, j]
    
    # Update x_cl only if k + 1 is within bounds
        if k + 1 < T:
            x_cl[:, k + 1] = sys.A @ x_cl[:, k] + sys.B @ u_cl[:, k]

        y_cl[:, k] = sys.C @ x_cl[:, k] + sys.D @ u_cl[:, k]
        y_cl_noise[:, k] = y_cl[:, k] * (1 + noise_max * (-1 + 2 * np.random.rand(p, 1))).flatten()

    # Set new initial conditions for the next iteration
        u_init = np.hstack([u_init[:, 1:], u_cl[:, k:k+1]])
        y_init = np.hstack([y_init[:, 1:], y_cl_noise[:, k:k+1].reshape(p, 1)])

        u_optimal = sol[N-L+1:N-L+1+m*L].flatten()
        y_predicted = sol[N-L+1+m*L:N-L+1+(m+p)*L].flatten()

        y_k = y_cl[:, k].reshape(-1, 1)  # Ensure y_k is (2,1)
        
    # update W:
        J_L_value = 0.5 * np.dot(sol.T, H @ sol) + np.dot(f.T, sol) + np.dot(W.T, Phi_ybar_val)  + np.dot(W.T, J_Phi_ybar_val @ (y_predicted[:2].reshape(-1, 1) - ybar[:2].reshape(-1, 1)))


        
        # Calculate the VFA cost term based on the current W and the observed output
        Phi_ybar_val = Phi_ybar(y_k)
        V_hat = W.conjugate().T @ Phi_ybar_val
        gradient = 2 * (J_L_value - V_hat) * Phi_ybar_val

        # Update W
        W_new = W + alpha * gradient
        if np.linalg.norm(W_new - W) > epsilon_W or np.linalg.norm(W_new - W)==0:
            W = W_new
            W_history.append(W_new.flatten().tolist())
        else:
            converged = True
            print("Convergence achieved.")
            break

    # Check the convergence flag and break the for loop if True

plt.figure(figsize=(12, 6))
for i in range(M):
    plt.plot([w[i] for w in W_history], label=f'W[{i}]')

plt.xlabel('Iteration')
plt.ylabel('Weight value')
plt.title('Behavior of W over time')
plt.legend()
plt.show()
plt.figure()

# Subplot for u_1
plt.subplot(2, 2, 1)
plt.plot(range(1, T + 1), u_cl[0, :], label='u_1')
plt.plot(range(1, T + 1), [u_term[0]] * T, label='u_{1,eq}')
plt.legend()

# Subplot for u_2
plt.subplot(2, 2, 2)
plt.plot(range(1, T + 1), u_cl[1, :], label='u_2')
plt.plot(range(1, T + 1), [u_term[1]] * T, label='u_{2,eq}')
plt.legend()

# Subplot for y_1
plt.subplot(2, 2, 3)
plt.plot(range(1, T + 1), y_cl[0, :], label='y_1')
plt.plot(range(1, T + 1), [y_term[0]] * T, label='y_{1,eq}')
plt.legend()

# Subplot for y_2
plt.subplot(2, 2, 4)
plt.plot(range(1, T + 1), y_cl[1, :], label='y_2')
plt.plot(range(1, T + 1), [y_term[1]] * T, label='y_{2,eq}')
plt.legend()

# Show the plot
plt.show()
