import numpy as np
from numba import njit
import matplotlib
matplotlib.use("Agg")   # disable GUI backend
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

# initialize gating vars

@njit
def init_state(V0):
    n0 = alphan(V0) / (alphan(V0) + betan(V0))
    m0 = alpham(V0) / (alpham(V0) + betam(V0))
    h0 = alphah(V0) / (alphah(V0) + betah(V0))
    w0 = 1.0 / (1.0 + np.exp(-V0 / 5.0))
    return n0, m0, h0, w0

# sweep function
@njit
def run_3d_sweep(noise_rates, uva_rate, GKHT_values, EL_values):

    dt = 0.001
    T_end = 200.0
    num_steps = int(T_end / dt)

    rates = np.zeros((len(noise_rates),
                      len(GKHT_values),
                      len(EL_values)))

    V_store = np.zeros((len(noise_rates),
                        len(GKHT_values),
                        len(EL_values),
                        num_steps))

    g_exc_store = np.zeros_like(V_store)

    t0 = 50.0
    t1 = 200.0

    tau_exc = constants()[13]
    tau_inh = constants()[14]

    for a, noise_rate in enumerate(noise_rates):
        for b, GKHT in enumerate(GKHT_values):
            for c, EL in enumerate(EL_values):

                V = EL
                n, m, h, w = init_state(V)
                g_exc = 0.0
                g_inh = 0.0

                dG_exc_uva = np.random.rand(5) * 0.5

                spike_count = 0
                t = 0.0

                refrac = 0
                V_prev = 0

                for k in range(num_steps):

                    lam_noise = noise_rate / 1000
                    lam_uva   = uva_rate / 1000

                    for i in range(5):
                        if np.random.rand() < lam_uva * dt:
                            g_exc += dG_exc_uva[i]

                    if np.random.rand() < lam_noise * dt:
                        g_exc += 0

                    V, n, m, h, w = RK4_step(V, n, m, h, w,
                                             g_exc, g_inh, dt,
                                             EL, GKHT)

                    g_exc *= np.exp(-dt / tau_exc)
                    g_inh *= np.exp(-dt / tau_inh)

                    t += dt

                    
                    V_store[a, b, c, k] = V
                    g_exc_store[a, b, c, k] = g_exc

                    
                    if t0 < t < t1:
                        THRESH = -20.0
                        REFRAC = int(2.0 / dt)   # 2 ms refractory

                        if refrac == 0:
                            if V_prev < THRESH and V >= THRESH:
                                spike_count += 1
                                refrac = REFRAC
                        else:
                            refrac -= 1
                    V_prev = V

                rates[a, b, c] = spike_count / ((t1 - t0) / 1000.0)

    return rates, V_store, g_exc_store

noise_exc_rates = [250]
uva_rate = 200
GKHT_values = np.linspace(100, 600, 10)
EL_values = [-75] #np.linspace(-100, -65, 4)

rates, V_store, g_exc_store = run_3d_sweep(noise_exc_rates, uva_rate, GKHT_values, EL_values)
num_exc = len(noise_exc_rates)

cols = min(2, num_exc)
rows = int(np.ceil(num_exc / cols))

fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))

axes = np.array(axes).reshape(-1)

for i, exc in enumerate(noise_exc_rates):
    ax = axes[i]

    heat = rates[i, :, :]

    im = ax.imshow(
        heat,
        origin='lower',
        aspect='auto',
        extent=[EL_values[0], EL_values[-1],
                GKHT_values[0], GKHT_values[-1]]
    )

    ax.set_title(f"Noise Rate = {exc} Hz")
    ax.set_xlabel("EL (mV)")
    ax.set_ylabel("GKHT")

    fig.colorbar(im, ax=ax, label="Firing Rate (Hz)")

plt.savefig("heatmap_all.png", dpi=150)
plt.close()

dt = 0.001
num_steps = V_store.shape[-1]
t = np.arange(num_steps) * dt

for a, noise_rate in enumerate(noise_exc_rates):

    rows = len(GKHT_values)
    cols = len(EL_values)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4 * cols, 3 * rows),
        sharex=True, sharey=True
    )

    axes = np.array(axes).reshape(-1)

    # Force axes into 2D array even if rows/cols = 1
    axes = np.array(axes).reshape(rows, cols)

    fig.suptitle(f"Voltage Traces for noise_rate = {noise_rate} Hz", fontsize=16)

    for b, GKHT in enumerate(GKHT_values):
        for c, EL in enumerate(EL_values):

            V = V_store[a, b, c, :]

            ax = axes[b, c]
            ax.plot(t, V, lw=1.0)

            if b == rows - 1:
                ax.set_xlabel(f"EL = {EL} mV")
            if c == 0:
                ax.set_ylabel(f"GKHT = {GKHT}")

            ax.grid(True)

plt.tight_layout()
plt.savefig(f"vtracegrid{noise_rate}.png", dpi=150)
plt.close()