import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# =========================
# Параметры задачи
# =========================
A = 3.0
B = 3.0
C = -3.0

c2_main = 0.05          # задано в варианте
c2_opp = 1.0            # метод (26): здесь поставлен классический Хойн
p_main = 2
p_opp = 2

x0, x1 = 0.0, 5.0
y0 = np.array([1.0, 1.0, A, 1.0], dtype=float)

tol = 1e-5
rtol = 1e-6
atol = 1e-12

EPS = 1e-300            # защита от log/степени при очень грубых шагах


# =========================
# Точное решение
# =========================
def exact_solution(x):
    s = np.sin(x * x)
    y1 = np.exp(s)
    y2 = np.exp(B * s)
    y3 = A + C * s
    y4 = np.cos(x * x)
    return np.array([y1, y2, y3, y4], dtype=float)


# =========================
# Правая часть системы
# =========================
def rhs(x, y):
    y1 = float(max(y[0], EPS))
    y2 = float(max(y[1], EPS))
    y3 = float(y[2])
    y4 = float(y[3])

    dy1 = 2.0 * x * (y2 ** (1.0 / B)) * y4
    dy2 = 2.0 * B * x * np.exp((B / C) * (y3 - A)) * y4
    dy3 = 2.0 * C * x * y4
    dy4 = -2.0 * x * np.log(y1)

    return np.array([dy1, dy2, dy3, dy4], dtype=float)


# =========================
# Явный двухэтапный RK2 семейства
# k1 = f(x_n, y_n)
# k2 = f(x_n + c2*h, y_n + c2*h*k1)
# y_{n+1} = y_n + h*(b1*k1 + b2*k2)
# b2 = 1/(2*c2), b1 = 1 - 1/(2*c2)
# =========================
def make_rk2_step(c2):
    b2 = 1.0 / (2.0 * c2)
    b1 = 1.0 - b2

    def step(x, y, h, k1=None):
        nfev = 0
        if k1 is None:
            k1 = rhs(x, y)
            nfev += 1
        k2 = rhs(x + c2 * h, y + c2 * h * k1)
        nfev += 1
        y_next = y + h * (b1 * k1 + b2 * k2)
        return y_next, nfev, k1

    return step


step_main = make_rk2_step(c2_main)
step_opp = make_rk2_step(c2_opp)


# =========================
# Фиксированный шаг
# =========================
def solve_fixed(step_fun, h, x0=x0, x1=x1, y0=y0):
    n = int(round((x1 - x0) / h))
    xs = np.linspace(x0, x1, n + 1)
    ys = np.zeros((n + 1, len(y0)), dtype=float)
    ys[0] = y0.copy()

    x = x0
    y = y0.copy()
    nfev = 0

    for i in range(n):
        y, c, _ = step_fun(x, y, h)
        nfev += c
        x = xs[i + 1]
        ys[i + 1] = y
        if not np.all(np.isfinite(y)):
            ys[i + 1:] = np.nan
            break

    return xs, ys, nfev


# =========================
# Норма полной погрешности
# =========================
def full_error_norm(y_num, x, ord_val=2):
    y_ex = exact_solution(x)
    return np.linalg.norm(y_ex - y_num, ord=ord_val)


# =========================
# График ошибки в конце отрезка от шага
# =========================
def endpoint_errors(step_fun, p, k_max=6):
    hs = []
    errs = []
    nfevs = []

    for k in range(k_max + 1):
        h = 1.0 / (2 ** k)
        xs_h, ys_h, nfev_h = solve_fixed(step_fun, h)
        xs_h2, ys_h2, nfev_h2 = solve_fixed(step_fun, h / 2.0)

        y_h = ys_h[-1]
        y_h2 = ys_h2[-1]

        if (np.any(~np.isfinite(y_h)) or np.any(~np.isfinite(y_h2))):
            err_end = np.nan
            runge_est = np.nan
        else:
            err_end = full_error_norm(y_h, x1, ord_val=2)
            runge_est = np.linalg.norm(y_h2 - y_h, ord=2) / (2 ** p - 1)

        hs.append(h)
        errs.append(err_end)
        nfevs.append((nfev_h, nfev_h2, runge_est))

    return np.array(hs), np.array(errs), nfevs


hs_main, errs_main, info_main = endpoint_errors(step_main, p_main, k_max=6)
hs_opp, errs_opp, info_opp = endpoint_errors(step_opp, p_opp, k_max=6)

# фильтрация для построения графика
mask_main = np.isfinite(errs_main) & (errs_main > 0)
mask_opp = np.isfinite(errs_opp) & (errs_opp > 0)

# Таблица по ошибкам в конце отрезка
rows = []
for i, h in enumerate(hs_main):
    rows.append({
        "method": f"main c2={c2_main}",
        "h": h,
        "||e(5)||_2": errs_main[i],
        "Runge est.": info_main[i][2],
    })
for i, h in enumerate(hs_opp):
    rows.append({
        "method": f"opp c2={c2_opp}",
        "h": h,
        "||e(5)||_2": errs_opp[i],
        "Runge est.": info_opp[i][2],
    })

df_errors = pd.DataFrame(rows)
print("\nОшибки в конце отрезка:")
print(df_errors.to_string(index=False))

plt.figure(figsize=(8, 6))
if np.any(mask_main):
    plt.loglog(hs_main[mask_main], errs_main[mask_main], "o-", label=f"method c2={c2_main}")
if np.any(mask_opp):
    plt.loglog(hs_opp[mask_opp], errs_opp[mask_opp], "s-", label=f"method (26), c2={c2_opp}")

# опорная прямая со склонением 2
ref_h = None
ref_e = None
if np.any(mask_main):
    ref_h = hs_main[mask_main][0]
    ref_e = errs_main[mask_main][0]
elif np.any(mask_opp):
    ref_h = hs_opp[mask_opp][0]
    ref_e = errs_opp[mask_opp][0]

if ref_h is not None:
    h_ref = np.array([min(hs_main.min(), hs_opp.min()), max(hs_main.max(), hs_opp.max())])
    e_ref = ref_e * (h_ref / ref_h) ** 2
    plt.loglog(h_ref, e_ref, "--", label="slope 2")

plt.xlabel("h")
plt.ylabel(r"$\|e(5)\|_2$")
plt.grid(True, which="both")
plt.legend()
plt.title("Error at x=5 vs step")
plt.tight_layout()
plt.show()


# =========================
# Поиск hopt по правилу Рунге
# =========================
def find_hopt(step_fun, p, tol, k_max=20):
    for k in range(k_max + 1):
        h = 1.0 / (2 ** k)
        xs_h, ys_h, _ = solve_fixed(step_fun, h)
        xs_h2, ys_h2, _ = solve_fixed(step_fun, h / 2.0)

        y_h = ys_h[-1]
        y_h2 = ys_h2[-1]

        if np.any(~np.isfinite(y_h)) or np.any(~np.isfinite(y_h2)):
            continue

        runge_est = np.linalg.norm(y_h2 - y_h, ord=2) / (2 ** p - 1)
        if runge_est <= tol:
            return h, k, runge_est

    raise RuntimeError("Не удалось найти hopt в заданном диапазоне k.")


hopt_main, kopt_main, est_main = find_hopt(step_main, p_main, tol)
hopt_opp, kopt_opp, est_opp = find_hopt(step_opp, p_opp, tol)

print(f"\nОптимальный шаг (main): hopt = {hopt_main}, k = {kopt_main}, Runge est = {est_main:.3e}")
print(f"Оптимальный шаг (opp) : hopt = {hopt_opp}, k = {kopt_opp}, Runge est = {est_opp:.3e}")

# Решение с hopt и график ошибки по x
def plot_solution_and_error(step_fun, hopt, title_prefix):
    xs, ys, nfev = solve_fixed(step_fun, hopt)
    exact_vals = np.array([exact_solution(x) for x in xs])
    err = np.linalg.norm(exact_vals - ys, axis=1)

    plt.figure(figsize=(9, 6))
    plt.plot(xs, ys[:, 0], label="y1")
    plt.plot(xs, ys[:, 1], label="y2")
    plt.plot(xs, ys[:, 2], label="y3")
    plt.plot(xs, ys[:, 3], label="y4")
    plt.plot(xs, exact_vals[:, 0], "--", alpha=0.6)
    plt.plot(xs, exact_vals[:, 1], "--", alpha=0.6)
    plt.plot(xs, exact_vals[:, 2], "--", alpha=0.6)
    plt.plot(xs, exact_vals[:, 3], "--", alpha=0.6)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title_prefix}: solution, hopt={hopt}, nfev={nfev}")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.semilogy(xs, err)
    plt.grid(True, which="both")
    plt.xlabel("x")
    plt.ylabel(r"$\|y_{exact}-y_h\|_2$")
    plt.title(f"{title_prefix}: full error for hopt={hopt}")
    plt.tight_layout()
    plt.show()

    return xs, ys, err, nfev

xs_main, ys_main, err_main, nfev_main = plot_solution_and_error(step_main, hopt_main, "Main method")
xs_opp, ys_opp, err_opp, nfev_opp = plot_solution_and_error(step_opp, hopt_opp, "Opponent method")


# =========================
# Адаптивный шаг по правилу Рунге
# =========================
def adaptive_solve(step_fun, p, rtol, atol, h0=1e-2):
    x = x0
    y = y0.copy()
    h = h0

    xs = [x]
    ys = [y.copy()]
    hs_accept = []
    x_accept = []

    hs_reject = []
    x_reject = []

    nfev = 0
    safety = 0.9
    min_fac = 0.2
    max_fac = 5.0

    while x < x1 - 1e-15:
        h = min(h, x1 - x)

        # Шаг h
        y_full, c1, k1 = step_fun(x, y, h)
        # Два шага h/2
        y_half1, c2, _ = step_fun(x, y, h / 2.0, k1=k1)
        y_half2, c3, _ = step_fun(x + h / 2.0, y_half1, h / 2.0)

        nfev += c1 + c2 + c3

        if (not np.all(np.isfinite(y_full))) or (not np.all(np.isfinite(y_half2))):
            # шаг явно плохой
            x_reject.append(x)
            hs_reject.append(h)
            h *= 0.5
            continue

        # Нормированная локальная ошибка по правилу Рунге
        err_vec = (y_half2 - y_full) / (2 ** p - 1)
        scale = atol + rtol * np.maximum(np.abs(y_half2), np.abs(y_full))
        err_norm = np.max(np.abs(err_vec) / scale)

        if err_norm <= 1.0:
            # принять
            x_accept.append(x)
            hs_accept.append(h)

            x = x + h
            y = y_half2.copy()

            xs.append(x)
            ys.append(y.copy())

            if err_norm == 0.0:
                fac = max_fac
            else:
                fac = safety * err_norm ** (-1.0 / (p + 1))

            fac = min(max_fac, max(min_fac, fac))
            h *= fac
        else:
            # отклонить
            x_reject.append(x)
            hs_reject.append(h)

            fac = safety * err_norm ** (-1.0 / (p + 1))
            fac = min(1.0, max(min_fac, fac))
            h *= fac

    return (
        np.array(xs),
        np.array(ys),
        np.array(x_accept),
        np.array(hs_accept),
        np.array(x_reject),
        np.array(hs_reject),
        nfev,
    )


def plot_adaptive(step_fun, p, rtol, atol, title_prefix, h0=1e-2):
    xs, ys, x_acc, h_acc, x_rej, h_rej, nfev = adaptive_solve(step_fun, p, rtol, atol, h0=h0)
    exact_vals = np.array([exact_solution(x) for x in xs])
    err = np.linalg.norm(exact_vals - ys, axis=1)

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys[:, 0], label="y1")
    plt.plot(xs, ys[:, 1], label="y2")
    plt.plot(xs, ys[:, 2], label="y3")
    plt.plot(xs, ys[:, 3], label="y4")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title_prefix}: adaptive solution, nfev={nfev}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    if len(h_acc) > 0:
        plt.scatter(x_acc, h_acc, s=25, marker="o", label="accepted")
    if len(h_rej) > 0:
        plt.scatter(x_rej, h_rej, s=35, marker="x", label="rejected")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("h")
    plt.title(f"{title_prefix}: step size vs x")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.semilogy(xs, err)
    plt.grid(True, which="both")
    plt.xlabel("x")
    plt.ylabel(r"$\|y_{exact}-y_h\|_2$")
    plt.title(f"{title_prefix}: full error")
    plt.tight_layout()
    plt.show()

    return xs, ys, err, nfev, len(h_acc), len(h_rej)


# адаптивный расчет для двух методов
xs_ad_main, ys_ad_main, err_ad_main, nfev_ad_main, nacc_main, nrej_main = plot_adaptive(
    step_main, p_main, rtol, atol, "Main method", h0=1e-2
)

xs_ad_opp, ys_ad_opp, err_ad_opp, nfev_ad_opp, nacc_opp, nrej_opp = plot_adaptive(
    step_opp, p_opp, rtol, atol, "Opponent method", h0=1e-2
)

print("\nAdaptive summary:")
print(f"main: nfev={nfev_ad_main}, accepted={nacc_main}, rejected={nrej_main}, final error={err_ad_main[-1]:.3e}")
print(f"opp : nfev={nfev_ad_opp}, accepted={nacc_opp}, rejected={nrej_opp}, final error={err_ad_opp[-1]:.3e}")


# =========================
# Число обращений к правой части от rtol
# =========================
rtols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

def nfev_vs_rtol(step_fun, p, label):
    data = []
    for r in rtols:
        _, _, _, nfev, nacc, nrej = plot_adaptive(step_fun, p, r, atol, f"{label}, rtol={r}", h0=1e-2)
        data.append((r, nfev, nacc, nrej))
    return pd.DataFrame(data, columns=["rtol", "nfev", "accepted", "rejected"])

# Для экономии времени, если нужно только итоговый график,
# можно закомментировать обе строки ниже и оставить один метод.
df_nfev_main = nfev_vs_rtol(step_main, p_main, "Main method")
df_nfev_opp = nfev_vs_rtol(step_opp, p_opp, "Opponent method")

print("\nNumber of RHS calls vs rtol (main):")
print(df_nfev_main.to_string(index=False))
print("\nNumber of RHS calls vs rtol (opponent):")
print(df_nfev_opp.to_string(index=False))

plt.figure(figsize=(8, 6))
plt.loglog(df_nfev_main["rtol"], df_nfev_main["nfev"], "o-", label="main")
plt.loglog(df_nfev_opp["rtol"], df_nfev_opp["nfev"], "s-", label="opponent")
plt.grid(True, which="both")
plt.xlabel("rtol")
plt.ylabel("nfev")
plt.title("RHS calls vs rtol")
plt.legend()
plt.tight_layout()
plt.show()