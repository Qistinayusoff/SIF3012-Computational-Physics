import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

alpha = 1
lam = 4
hbar = 1
m = 1

def potential(x):
    return alpha**2 * lam * (lam - 1) * (0.5 - 1 / np.cosh(alpha * x)**2)

def numerov(x, psi0, psi1, E, V):
    dx = x[1] - x[0]
    psi = np.zeros_like(x)
    psi[0], psi[1] = psi0, psi1

    for i in range(1, len(x) - 1):
        k1 = 2 * (E - V(x[i-1]))
        k2 = 2 * (E - V(x[i]))
        k3 = 2 * (E - V(x[i+1]))
        psi[i+1] = (2 * (1 - 5 * dx**2 / 12 * k2) * psi[i] - 
                    (1 + dx**2 / 12 * k1) * psi[i-1]) / (1 + dx**2 / 12 * k3)
    return psi

def matching_function(E, x, V):
    psiL = numerov(x, 0, 1e-3, E, V)
    psiR = numerov(x[::-1], 0, 1e-3, E, V)[::-1]
    midpoint = len(x) // 2
    if psiL[midpoint] == 0 or psiR[midpoint] == 0:  
        return np.inf
    return (psiL[midpoint] / psiL[midpoint - 1]) - (psiR[midpoint] / psiR[midpoint - 1])

def find_eigenvalues(x, V, n_states):
    eigenvalues = []
    for n in range(n_states):
        E_min, E_max = n + 0.1, n + 1.5
        try:
            result = root_scalar(matching_function, args=(x, V), bracket=(E_min, E_max))
            if result.converged:
                eigenvalues.append(result.root)
        except ValueError:
            print(f"Warning: Unable to find eigenvalue for n = {n}")
    return eigenvalues

x = np.linspace(-5, 5, 1000)  
V = lambda x: potential(x)     
n_states = 3                 

eigenvalues = find_eigenvalues(x, V, n_states)

plt.figure(figsize=(10, 6))
for E in eigenvalues:
    psi = numerov(x, 0, 1e-3, E, V)
    psi /= np.sqrt(np.sum(psi**2) * (x[1] - x[0]))  
    plt.plot(x, psi, label=f"E = {E:.4f}")

plt.plot(x, potential(x), label="Potential V(x)", linestyle="-", color="green")
plt.xlabel("x")
plt.ylabel("Psi(x) and V(x)")
plt.title("Wavefunctions and Potential")
plt.legend()
plt.grid()
plt.show()

print("Eigenvalues (E):", eigenvalues)
