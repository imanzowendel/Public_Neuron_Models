import numpy as np
from numba import njit
import matplotlib.pyplot as plt

# cpnstants 

@njit
def constants():
    ENa  = 55.0
    EK   = -80.0
    EI   = -75.0
    EE   = 0.0

    GL   = 0.1
    GNa  = 100.0
    GKdr = 20.0

    Cm   = 1.0
    A    = 6000.0

    tau_n   = 1.0
    tau_m   = 1.0
    tau_h   = 1.0
    tau_w   = 1.0
    tau_exc = 2.0
    tau_inh = 5.0

    dG_exc = 0.5
    dG_inh = 0.03

    return (
        ENa, EK, EI, EE,
        GL, GNa, GKdr,
        Cm, A,
        tau_n, tau_m, tau_h, tau_w, tau_exc, tau_inh,
        dG_exc, dG_inh
    )

# gating functions

@njit
def alphan(V):
    return 0.15 * (V + 15.0) / (1.0 - np.exp(-(V + 15.0) / 10.0))

@njit
def betan(V):
    return 0.2 * np.exp(-(V + 25.0) / 80.0)

@njit
def alpham(V):
    return (V + 22.0) / (1.0 - np.exp(-(V + 22.0) / 10.0))

@njit
def betam(V):
    return 40.0 * np.exp(-(V + 47.0) / 18.0)

@njit
def alphah(V):
    return 0.7 * np.exp(-(V + 34.0) / 20.0)

@njit
def betah(V):
    return 10.0 / (1.0 + np.exp(-(V + 4.0) / 10.0))

@njit
def dx_dt(V, x, alpha, beta):
    return alpha(V) * (1.0 - x) - beta(V) * x

@njit
def dw_dt(V, w):
    (_, _, _, _,
     _, _, _,
     _, _,
     _, _, _, tau_w, _, _,
     _, _) = constants()
    w_inf = 1.0 / (1.0 + np.exp(-V / 5.0))
    return (w_inf - w) / tau_w

# standard voltage dynamics function

@njit
def dV_dt(V, n, m, h, w, g_exc, g_inh, EL, GKHT):
    (
        ENa, EK, EI, EE,
        GL, GNa, GKdr,
        Cm, A,
        tau_n, tau_m, tau_h, tau_w, tau_exc, tau_inh,
        dG_exc, dG_inh
    ) = constants()

    IL   = -GL   * (V - EL)
    IKdr = -GKdr * (n**4) * (V - EK)
    IKHT = -GKHT * w * (V - EK)
    INa  = -GNa  * (m**3) * h * (V - ENa)
    Iexc = -g_exc * (V - EE)
    Iinh = -g_inh * (V - EI)

    return (IL + IKdr + IKHT + INa + Iexc + Iinh) / Cm

# rk4 function

@njit
def RK4_step(V, n, m, h, w, g_exc, g_inh, dt, EL, GKHT):

    k1_V = dV_dt(V, n, m, h, w, g_exc, g_inh, EL, GKHT)
    k1_n = dx_dt(V, n, alphan, betan)
    k1_m = dx_dt(V, m, alpham, betam)
    k1_h = dx_dt(V, h, alphah, betah)
    k1_w = dw_dt(V, w)

    V2 = V + 0.5 * dt * k1_V
    n2 = n + 0.5 * dt * k1_n
    m2 = m + 0.5 * dt * k1_m
    h2 = h + 0.5 * dt * k1_h
    w2 = w + 0.5 * dt * k1_w

    k2_V = dV_dt(V2, n2, m2, h2, w2, g_exc, g_inh, EL, GKHT)
    k2_n = dx_dt(V2, n2, alphan, betan)
    k2_m = dx_dt(V2, m2, alpham, betam)
    k2_h = dx_dt(V2, h2, alphah, betah)
    k2_w = dw_dt(V2, w2)

    V3 = V + 0.5 * dt * k2_V
    n3 = n + 0.5 * dt * k2_n
    m3 = m + 0.5 * dt * k2_m
    h3 = h + 0.5 * dt * k2_h
    w3 = w + 0.5 * dt * k2_w

    k3_V = dV_dt(V3, n3, m3, h3, w3, g_exc, g_inh, EL, GKHT)
    k3_n = dx_dt(V3, n3, alphan, betan)
    k3_m = dx_dt(V3, m3, alpham, betam)
    k3_h = dx_dt(V3, h3, alphah, betah)
    k3_w = dw_dt(V3, w3)

    V4 = V + dt * k3_V
    n4 = n + dt * k3_n
    m4 = m + dt * k3_m
    h4 = h + dt * k3_h
    w4 = w + dt * k3_w

    k4_V = dV_dt(V4, n4, m4, h4, w4, g_exc, g_inh, EL, GKHT)
    k4_n = dx_dt(V4, n4, alphan, betan)
    k4_m = dx_dt(V4, m4, alpham, betam)
    k4_h = dx_dt(V4, h4, alphah, betah)
    k4_w = dw_dt(V4, w4)

    V_new = V + (dt/6)*(k1_V + 2*k2_V + 2*k3_V + k4_V)
    n_new = n + (dt/6)*(k1_n + 2*k2_n + 2*k3_n + k4_n)
    m_new = m + (dt/6)*(k1_m + 2*k2_m + 2*k3_m + k4_m)
    h_new = h + (dt/6)*(k1_h + 2*k2_h + 2*k3_h + k4_h)
    w_new = w + (dt/6)*(k1_w + 2*k2_w + 2*k3_w + k4_w)

    return V_new, n_new, m_new, h_new, w_new

# sweep function

@njit
def run_3d_sweep():

    exc_rates   = np.linspace(50, 400, 20)  
    GKHT_values = np.linspace(0, 1000, 20)
    EL_values   = np.linspace(0, -200.0, 20)

    dt = 0.001
    T_end = 200.0
    num_steps = int(T_end / dt)

    rates = np.zeros((exc_rates.size, GKHT_values.size, EL_values.size))

    t0 = 50.0
    t1 = 200.0

    THRESH = -20.0
    REFRAC_MS = 2.0
    REFRAC_STEPS = int(REFRAC_MS / dt)

    for a in range(exc_rates.size):
        exc_rate = exc_rates[a]

        for b in range(GKHT_values.size):
            GKHT = GKHT_values[b]

            for c in range(EL_values.size):
                EL = EL_values[c]

                V = -65.8
                n = 0.125
                m = 0.0048
                h = 9.1e-5
                w = 1.9e-6
                g_exc = 0.0
                g_inh = 0.0

                t = 0.0
                spike_count = 0
                V_prev = V
                refrac_count = 0

                for k in range(num_steps):

                    lam = exc_rate / 1000.0
                    if t >= t0 and t <= t1:
                        if np.random.rand() < lam * dt:
                            g_exc += 0.5

                    V, n, m, h, w = RK4_step(V, n, m, h, w, g_exc, g_inh, dt, EL, GKHT)
                    t += dt

                    if t0 < t < t1:
                        if refrac_count == 0:
                            if V_prev < THRESH and V >= THRESH:
                                spike_count += 1
                                refrac_count = REFRAC_STEPS
                        else:
                            refrac_count -= 1

                    V_prev = V

                rates[a, b, c] = spike_count / ((t1 - t0) / 1000.0)

    return exc_rates, GKHT_values, EL_values, rates

# running, plotting and saving sweep results

exc_rates, GKHT_values, EL_values, rates = run_3d_sweep()

print("3D sweep complete.")
print("rates.shape =", rates.shape)

rows = []
for i in range(len(exc_rates)):
    for j in range(len(GKHT_values)):
        for k in range(len(EL_values)):
            rows.append([
                exc_rates[i],
                GKHT_values[j],
                EL_values[k],
                rates[i, j, k]
            ])

np.savetxt("sweep_results.csv",
           rows,
           delimiter=",",
           header="exc_rate,GKHT,EL,firing_rate",
           comments="")

print("Saved sweep_results.csv")

num_exc = len(exc_rates)
rows = int(np.ceil(num_exc / 2))
cols = 2

fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
axes = axes.flatten()

for i in range(num_exc):
    ax = axes[i]

    heat = rates[i, :, :]   # shape: [GKHT, EL]

    im = ax.imshow(heat, origin='lower', aspect='auto',
                   extent=[EL_values[0], EL_values[-1],
                           GKHT_values[0], GKHT_values[-1]])

    ax.set_title(f"exc_rate = {exc_rates[i]} Hz")
    ax.set_xlabel("EL (mV)")
    ax.set_ylabel("GKHT")

    fig.colorbar(im, ax=ax, label="Firing Rate (Hz)")

for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

def simulate_trace(exc_rate, GKHT_default=1500.0, EL_default=-20.0):
    dt = 0.001
    T_end = 200.0
    num_steps = int(T_end / dt)

    V_trace = np.zeros(num_steps)
    t_trace = np.zeros(num_steps)

    # Initial conditions
    V = -65.8
    n = 0.125
    m = 0.0048
    h = 9.1e-5
    w = 1.9e-6
    g_exc = 0.0
    g_inh = 0.0

    t = 0.0

    for k in range(num_steps):

        # Poisson excitation
        lam = exc_rate / 1000.0
        if np.random.rand() < lam * dt:
            g_exc += 0.5

        V, n, m, h, w = RK4_step(V, n, m, h, w, g_exc, g_inh, dt, EL_default, GKHT_default)
        t += dt

        V_trace[k] = V
        t_trace[k] = t

    return t_trace, V_trace


def simulate_trace(exc_rate, GKHT, EL):
    dt = 0.001
    T_end = 200.0
    num_steps = int(T_end / dt)

    V_trace = np.zeros(num_steps)
    t_trace = np.zeros(num_steps)

    # Initial conditions
    V = -65.0
    n = alphan(V) / (alphan(V) + betan(V))
    m = alpham(V) / (alpham(V) + betam(V))
    h = alphah(V) / (alphah(V) + betah(V))
    w = 1.0 / (1.0 + np.exp(-V / 5.0))
    g_exc = 0.0
    g_inh = 0.0

    t = 0.0

    for k in range(num_steps):

        # Poisson excitation
        lam = exc_rate / 1000.0
        if np.random.rand() < lam * dt:
            g_exc += 0.5

        V, n, m, h, w = RK4_step(V, n, m, h, w, g_exc, g_inh, dt, EL, GKHT)
        t += dt

        V_trace[k] = V
        t_trace[k] = t * 1000.0   # convert to ms

    return t_trace, V_trace


# for plotting v vs t
"""
for exc in exc_rates:

    rows = len(GKHT_values)
    cols = len(EL_values)

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharex=True, sharey=True)
    fig.suptitle(f"Voltage Traces for exc_rate = {exc} Hz", fontsize=16)

    # Ensure axes is 2D even if rows/cols = 1
    axes = np.atleast_2d(axes)

    for i, GKHT in enumerate(GKHT_values):
        for j, EL in enumerate(EL_values):

            t, V = simulate_trace(exc, GKHT, EL)

            ax = axes[i, j]
            ax.plot(t, V, lw=1.0)

            if i == rows - 1:
                ax.set_xlabel(f"EL = {EL} mV")
            if j == 0:
                ax.set_ylabel(f"GKHT = {GKHT}")

            ax.grid(True)

    plt.tight_layout()
    plt.show()
"""