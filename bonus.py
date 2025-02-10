import numpy as np
import matplotlib.pyplot as plt


def sir_model_rk4(S0, I0, R0, beta, gamma, h, days):
    def dSdt(S, I):
        return -beta * S * I

    def dIdt(S, I):
        return beta * S * I - gamma * I

    def dRdt(I):
        return gamma * I

    steps = int(days / h) + 1  # Исправлено: включаем последний шаг
    t = np.arange(0, days + h, h)  # Исправлено: корректное формирование временной оси
    S, I, R = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    S[0], I[0], R[0] = S0, I0, R0

    for n in range(steps - 1):
        k1S = h * dSdt(S[n], I[n])
        k1I = h * dIdt(S[n], I[n])
        k1R = h * dRdt(I[n])

        k2S = h * dSdt(S[n] + k1S / 2, I[n] + k1I / 2)
        k2I = h * dIdt(S[n] + k1S / 2, I[n] + k1I / 2)
        k2R = h * dRdt(I[n] + k1I / 2)

        k3S = h * dSdt(S[n] + k2S / 2, I[n] + k2I / 2)
        k3I = h * dIdt(S[n] + k2S / 2, I[n] + k2I / 2)
        k3R = h * dRdt(I[n] + k2I / 2)

        k4S = h * dSdt(S[n] + k3S, I[n] + k3I)
        k4I = h * dIdt(S[n] + k3S, I[n] + k3I)
        k4R = h * dRdt(I[n] + k3I)

        S[n + 1] = S[n] + (k1S + 2 * k2S + 2 * k3S + k4S) / 6
        I[n + 1] = I[n] + (k1I + 2 * k2I + 2 * k3I + k4I) / 6
        R[n + 1] = R[n] + (k1R + 2 * k2R + 2 * k3R + k4R) / 6

    return t, S, I, R


# Начальные условия
S0, I0, R0 = 999000, 1000, 0
beta, gamma = 0.0003, 0.1
h, days = 0.1, 100

t, S, I, R = sir_model_rk4(S0, I0, R0, beta, gamma, h, days)

# Пиковое число зараженных и день пика
peak_I = max(I)
peak_day = t[np.argmax(I)]
total_infected = max(1_000_000 - S)  # Исправлено: максимальное количество зараженных

# Построение графиков
plt.figure(figsize=(10, 5))
plt.plot(t, S, label="Susceptible")
plt.plot(t, I, label="Infected")
plt.plot(t, R, label="Recovered")
plt.axvline(peak_day, color='r', linestyle='--', label=f'Peak: Day {peak_day:.1f}')
plt.xlabel("Days")
plt.ylabel("Population")
plt.title("SIR Model Simulation (RK4)")
plt.legend()
plt.grid()
plt.ylim(0, 1_000_000)  # Исправлено: масштаб графика
plt.show()

print(f"Peak Infected: {peak_I:.0f} on Day {peak_day:.1f}")
print(f"Total Infected During Outbreak: {total_infected:.0f}")


# Численное интегрирование для энергопотребления

def trapezoidal_rule(x, y):
    return np.trapezoid(y, x)


def simpsons_rule(x, y):
    if len(x) % 2 == 0:  # Исправлено: метод Симпсона требует нечетного числа точек
        x = np.append(x, x[-1] + (x[-1] - x[-2]))
        y = np.append(y, y[-1])
    return np.sum((x[2::2] - x[:-2:2]) / 6 * (y[:-2:2] + 4 * y[1::2] + y[2::2]))


# Данные энергопотребления
time = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
power = np.array([500, 480, 450, 600, 800, 950, 1000, 980, 920, 850, 700, 550, 500])

energy_trap = trapezoidal_rule(time, power)
energy_simp = simpsons_rule(time, power)

print(f"Total Energy Consumption (Trapezoidal Rule): {energy_trap:.2f} MWh")
print(f"Total Energy Consumption (Simpson's Rule): {energy_simp:.2f} MWh")
