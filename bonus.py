import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi


def sir_model(t, S, I, R, beta, gamma):
    if np.isnan(S) or np.isnan(I) or np.isnan(R) or np.isinf(S) or np.isinf(I) or np.isinf(R):
        return np.array([0.0, 0.0, 0.0])  # Останавливаем распространение NaN

    dSdt = -beta * S * I
    dIdt = min(beta * S * I - gamma * I, 1e6)  # Ограничиваем рост I
    dRdt = gamma * I

    return np.array([dSdt, dIdt, dRdt])



def runge_kutta4(f, S0, I0, R0, beta, gamma, h, days):
    steps = int(days / h)
    S, I, R = np.zeros(steps + 1), np.zeros(steps + 1), np.zeros(steps + 1)
    S[0], I[0], R[0] = S0, I0, R0

    for n in range(steps):
        t = n * h
        k1 = h * f(t, S[n], I[n], R[n], beta, gamma)
        k2 = h * f(t + h / 2, S[n] + k1[0] / 2, I[n] + k1[1] / 2, R[n] + k1[2] / 2, beta, gamma)
        k3 = h * f(t + h / 2, S[n] + k2[0] / 2, I[n] + k2[1] / 2, R[n] + k2[2] / 2, beta, gamma)
        k4 = h * f(t + h, S[n] + k3[0], I[n] + k3[1], R[n] + k3[2], beta, gamma)

        S[n + 1] = S[n] + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        I[n + 1] = I[n] + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        R[n + 1] = R[n] + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6

    return S, I, R


# Initial conditions
S0, I0, R0 = np.float64(999000), np.float64(1000), np.float64(0)
beta, gamma = 0.0003, 0.1
h, days = 0.05, 100

# Solve original scenario
S, I, R = runge_kutta4(sir_model, S0, I0, R0, beta, gamma, h, days)

days_range = np.linspace(0, days, len(S))
peak_infected = np.max(I)
peak_day = days_range[np.argmax(I)]
total_infected = S0 - S[-1]

# Vaccination scenario (50% reduction in susceptible population)
S_v, I_v, R_v = runge_kutta4(sir_model, S0 * 0.5, I0, R0, beta, gamma, h, days)

# Social distancing scenario (50% reduction in beta)
S_sd, I_sd, R_sd = runge_kutta4(sir_model, S0, I0, R0, beta * 0.5, gamma, h, days)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(days_range, S, label='Susceptible', color='blue')
plt.plot(days_range, I, label='Infected', color='red')
plt.plot(days_range, R, label='Recovered', color='green')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.grid()
plt.show()

# Comparisons for vaccination and social distancing
plt.figure(figsize=(10, 6))
plt.plot(days_range, I, label='Original Infected', color='red')
plt.plot(days_range, I_v, label='Vaccination 50% Infected', color='purple', linestyle='dashed')
plt.plot(days_range, I_sd, label='Social Distancing 50% Infected', color='orange', linestyle='dashed')
plt.xlabel('Days')
plt.ylabel('Infected Population')
plt.title('Effect of Interventions')
plt.legend()
plt.grid()
plt.show()

print(f'Peak infected: {peak_infected:.0f} on day {peak_day:.1f}')
print(f'Total infected at some point: {total_infected:.0f}')

# Given data
time = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
power = np.array([500, 480, 450, 600, 800, 950, 1000, 980, 920, 850, 700, 550, 500])

# Trapezoidal Rule
energy_trap = np.trapezoid(power, time)

# Simpson’s Rule
energy_simp = spi.simpson(power, time)

print(f"Total energy consumption using Trapezoidal Rule: {energy_trap:.2f} MWh")
print(f"Total energy consumption using Simpson's Rule: {energy_simp:.2f} MWh")