import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a, b = 1.5, 3.3
I = 7.25800298437456290561  # точное значение интеграла

def f(x):
    # f(x) = 2 cos(2.5x) exp(x/3) + 4 sin(3.5x) exp(−3x) + x
    return 2 * np.cos(2.5 * x) * np.exp(x / 3) + 4 * np.sin(3.5 * x) * np.exp(-3 * x) + x

alpha = 1/3
beta = 0
def weight(x):
    # p(x) = (x-a)^{-alpha} * (b-x)^{-beta}
    # Используем небольшое смещение eps, чтобы избежать деления на 0 в точке x = a (или x = b)
    eps = 1e-14
    return (x - a + eps)**(-alpha) * (b - x + eps)**(-beta)

rtol = 1e-10

# 1) Метод Симпсона для моментов веса
def simpson_integral(func, a, b, n=400):
    # Для формулы Симпсона число шагов четное
    if n % 2 == 1:
        n += 1

    x = np.linspace(a, b, n + 1)
    y = func(x)
    h = (b - a) / n
    return (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))

def moment_weighted(k, A, B):
    # μ_k = ∫_A^B x^k p(x) dx
    return simpson_integral(lambda x: (x**k) * weight(x), A, B, n=400)



# 2) Трёхточечная формула Ньютона–Котса на отрезке [a, b]
#    Узлы: a, (a + b)/2, b
#    Коэффициенты зависят от веса p(x)
def nc3_local(a, b):
    h = b - a
    x0 = a
    x1 = (a + b) / 2
    x2 = b

    # Базис Лагранжа на t in [0,1], x = a + h * t
    # l0(t) = 2t^2 - 3t + 1
    # l1(t) = -4t^2 + 4t
    # l2(t) = 2t^2 - t
    #
    # Весовые коэффициенты:
    # a_i = ∫_A^B p(x) l_i((x-A)/h) dx
    #
    # Перепишем l_i через x:
    # t = (x - a)/h
    def l0(x):
        t = (x - a) / h
        return 2*t*t - 3*t + 1

    def l1(x):
        t = (x - a) / h
        return -4*t*t + 4*t

    def l2(x):
        t = (x - a) / h
        return 2*t*t - t

    a0 = simpson_integral(lambda x: weight(x) * l0(x), a, b, n=300)
    a1 = simpson_integral(lambda x: weight(x) * l1(x), a, b, n=300)
    a2 = simpson_integral(lambda x: weight(x) * l2(x), a, b, n=300)

    return a0 * f(x0) + a1 * f(x1) + a2 * f(x2)


# 3) Трёхточечная формула Гаусса-Лежандра на отрезке [a, b]
# ============================================================
def gauss3_local(a, b):
    h = b - a
    m = (a + b) / 2

    t = np.sqrt(3 / 5) * h / 2

    x1 = m - t
    x2 = m
    x3 = m + t

    return (h / 2) * (
        (5 / 9) * f(x1) +
        (8 / 9) * f(x2) +
        (5 / 9) * f(x3)
    )


# 4) Составные формулы
def composite_nc3(N):
    # Составная трёхточечная Ньютона–Котса
    x = np.linspace(a, b, N + 1)
    S = 0.0
    for i in range(N):
        S += nc3_local(x[i], x[i + 1])
    return S

def composite_gauss3(N):
    # Составная трёхточечная Гаусса
    x = np.linspace(a, b, N + 1)
    S = 0.0
    for i in range(N):
        S += gauss3_local(x[i], x[i + 1])
    return S


# 5) Оценка p по Эйткену
def aitken_p(S1, S2, S3):
    num = abs(S3 - S2)
    den = abs(S2 - S1)
    if num == 0 or den == 0:
        return np.nan
    return np.log2(den / num)

# 6) Построение таблицы
def build_table(method, method_name, rtol, p_theory):
    rows = []

    S_vals = []
    Sr_vals = []

    N = 1

    while True:
        h = (b - a) / N
        S = method(N)

        S_vals.append(S)

        row = {
            "N": N,
            "h": h,
            "S": S,
            "E": abs(I - S) / abs(I),
            "R": np.nan,
            "p": np.nan,
            "S'": np.nan,
            "E'": np.nan,
            "p'": np.nan,
            "pr": np.nan
        }


        if len(S_vals) >= 2:
            # Для оценки Рунге используем p из предыдущей аппроксимации,
            # если оно уже стабилизировалось
            #
            # Проверка по Эйткену
            if len(S_vals) >= 3:
                p_est = aitken_p(
                    S_vals[-3],
                    S_vals[-2],
                    S_vals[-1]
                )

                row["p"] = p_est
            else:
                p_est = np.nan


            # Используем теоретический p (p_theory),
            # пока p не стабилизуется
            if np.isnan(p_est):
                p_used = p_theory
            else:
                p_used = p_est

            S_prev = S_vals[-2]

            delta = (S - S_prev) / (2**p_used - 1)

            R = abs(delta) / abs(S)

            S_rich = S + delta

            row["R"] = R
            row["S'"] = S_rich
            row["E'"] = abs(I - S_rich) / abs(I)

            Sr_vals.append(S_rich)

            # p' for значений Ричардсона
            if len(Sr_vals) >= 3:
                row["p'"] = aitken_p(
                    Sr_vals[-3],
                    Sr_vals[-2],
                    Sr_vals[-1]
                )

            # Наклон pr
            if len(rows) >= 1:

                R_prev = rows[-1]["R"]
                h_prev = rows[-1]["h"]

                if (
                    not np.isnan(R_prev)
                    and R_prev > 0
                    and R > 0
                ):
                    row["pr"] = (np.log10(R) - np.log10(R_prev)) / (np.log10(h) - np.log10(h_prev))

        rows.append(row)

        # Условие остановки по Рунге
        if not np.isnan(row["R"]) and row["R"] < rtol:
            break

        N *= 2

    df = pd.DataFrame(rows)

    print(f"\n{method_name}")

    pd.set_option("display.float_format", "{:.12e}".format)

    print(df.to_string(index=False))

    return df


# 7) График log10(R) от log10(h)
def plot_R(df, title, slope1, slope2=None):
    d = df.dropna(subset=["R", "h"]).copy()
    h = d["h"].to_numpy()
    R = d["R"].to_numpy()

    x = np.log10(h)
    y = np.log10(R)

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, "o-", label="log10(R)")

    # Опорная прямая со наклоном slope1
    c1 = np.median(R / (h ** slope1))
    h_line = np.array([h.min(), h.max()])
    plt.plot(np.log10(h_line), np.log10(c1 * h_line**slope1),
             "k--", label=f"опорная прямая, наклон {slope1}")

    if slope2 is not None:
        c2 = np.median(R / (h ** slope2))
        plt.plot(np.log10(h_line), np.log10(c2 * h_line**slope2),
                 "k--", alpha=0.6, label=f"опорная прямая, наклон {slope2}")

    plt.xlabel("log10(h)")
    plt.ylabel("log10(R)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# 8) Подбор Nopt и проверка точности
def find_nopt(method, df, p_min, p_max, is_gauss=False):
    # Берём первую строку, где p попал в требуемый диапазон
    good = df[(df["p"].notna()) & (df["p"] > p_min) & (df["p"] < p_max)].copy()
    if good.empty:
        return None

    row = good.iloc[0]
    N0 = int(row["N"])
    R0 = float(row["R"])
    p0 = float(row["p"])

    # Оценка Nopt по Рунге
    N_est = int(np.ceil(N0 * (R0 / rtol) ** (1.0 / p0)))

    # Для симметричных формул N должен быть кратен 2, но здесь N — число отрезков,
    # а в нашем построении каждый отрезок считается отдельно, так что ограничение не нужно

    # Увеличиваем, пока точная погрешность не станет достаточной
    N = max(N_est, N0 + 1)
    while True:
        S = method(N)
        E = abs(I - S) / abs(I)
        if E < rtol:
            return {
                "Nrow": N0,
                "p": p0,
                "N_est": N_est,
                "Nopt": N,
                "E": E
            }
        N += 1


# 9)
nc_df = build_table(
    composite_nc3,
    "NC3",
    rtol,
    p_theory=4
)

gauss_df = build_table(
    composite_gauss3,
    "Gauss3",
    rtol,
    p_theory=6
)

# Графики
plot_R(nc_df, "Newton-Cotes 3-point: log10(R) vs log10(h)", 3, 4)
plot_R(gauss_df, "Gauss 3-point: log10(R) vs log10(h)", 6)

# Подбор Nopt
nc_opt = find_nopt(composite_nc3, nc_df, 2.7, 4.0)
gauss_opt = find_nopt(composite_gauss3, gauss_df, 5.7, 6.7, is_gauss=True)

print("\nNopt results")
print("NC3   :", nc_opt)
print("Gauss :", gauss_opt)