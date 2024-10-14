import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert


def hilbert_transform(signal):
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)

    return analytic_signal, envelope


# Define the wave function
def ripple(T, F, omega_t, omega_f, phase=0, direction=1):
    ripple = np.cos(2 * np.pi * (omega_t * T + direction* omega_f * F) + phase)
    ripple /= np.max(ripple)
    return ripple


def another_ripple(t, f, omega_t, omega_f, phase_t=0, phase_f=0, direction=1):
    h_t = np.cos(2 * np.pi * (omega_t * t) + phase_t)
    h_f = np.cos(2 * np.pi * (omega_f * f) + phase_f)

    h_t_a = hilbert(h_t)
    print('h_t_a : ',h_t_a.shape)
    h_f_a = hilbert(h_f)
    print('h_f_a : ',h_f_a.shape)

    strf_1 = h_t_a.reshape(1, len(t)) * h_f_a.reshape(len(f), 1)
    print(strf_1.shape)
    strf_2 = np.conj(h_t_a).reshape(1, len(t)) * h_f_a.reshape(len(f), 1)
    return strf_1.real/np.max(strf_1.real), strf_2.real/np.max(strf_2.real)

# def STRF_original(T, F, ):

if __name__ == '__main__':
    # Define the grid and initial time
    t = np.linspace(0, 200, 400)
    f = np.linspace(0, 8000, 128)

    omega_t = 64
    omega_f = 8

    T, F = np.meshgrid(t, f)
    STRF1, STRF2 = another_ripple(t, f, omega_t, omega_f)
    plt.figure()
    plt.contourf(T, F, STRF1, levels=100, cmap='viridis')
    plt.colorbar()
    plt.title("STRF 1")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

    plt.figure()
    plt.contourf(T, F, STRF2, levels=100, cmap='viridis')
    plt.colorbar()
    plt.title("STRF 2")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

    Z = ripple(T, F, omega_t=64, omega_f=8, phase=0, direction=1)
    plt.figure()
    plt.contourf(T, F, Z, levels=100, cmap='viridis')
    plt.colorbar()
    plt.title("Ripple Effect")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
