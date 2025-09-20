import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def pseudo_second_order(t, q, k2, qe):
    """
    Defines the pseudo-second-order kinetic model.
    t: time (unused, but required by solve_ivp)
    q: amount adsorbed at time t (mg/g)
    k2: pseudo-second-order rate constant (g/mg·min)
    qe: equilibrium adsorption capacity (mg/g)
    """
    return k2 * (qe - q)**2

# --- Simulation Parameters ---
k2 = 0.01  # g/mg·min, a hypothetical rate constant
qe = 50    # mg/g, a hypothetical equilibrium adsorption capacity
q0 = 0     # mg/g, initial amount adsorbed at time t=0
time_span = [0, 100] # min, simulation time from 0 to 100
t_eval = np.linspace(time_span[0], time_span[1], 500) # points to evaluate the solution

# --- Run the Simulation ---
solution = solve_ivp(
    fun=pseudo_second_order,
    t_span=time_span,
    y0=[q0],
    args=(k2, qe),
    t_eval=t_eval
)

# --- Plot the Results ---
if solution.success:
    q_t_simulated = solution.y[0]
    time_simulated = solution.t

    plt.figure(figsize=(10, 6))
    plt.plot(time_simulated, q_t_simulated, label='Simulated Adsorption Curve', color='teal', linewidth=2)
    plt.axhline(y=qe, color='red', linestyle='--', label=f'Equilibrium Capacity ($q_e$ = {qe} mg/g)')
    plt.title('Pseudo-Second-Order Adsorption Kinetics Simulation 🧪', fontsize=16)
    plt.xlabel('Time (min)', fontsize=12)
    plt.ylabel('Amount Adsorbed ($q_t$, mg/g)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Simulation successful. Final amount adsorbed: {q_t_simulated[-1]:.2f} mg/g")
else:
    print("Simulation failed.")