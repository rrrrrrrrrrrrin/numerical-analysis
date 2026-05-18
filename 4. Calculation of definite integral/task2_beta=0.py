import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Код решает задачу численного нтегрирования функции с весом,
# имеющим особенность (alpha или beta = 0) на конце отрезка (a или b)
#
# Внутри каждого подотрезка происходит аналитическое (точное) интегрирование веса
# Так можно обойти деление на ноль в точке a или b (вес p(x) = (x-a)^{-alpha} * (b-x)^{-beta})
# и сохранить высокую точность методов Ньютона-Котса и Гаусса

# Границы интегрирования и параметры степенных особенностей веса
a, b = 1.5, 3.3
alpha = 1.0 / 3.0  # Особенность (x-a)^(-1/3) на левом конце отрезка интегрирования [a, b]
beta = 0.0


# Интегрируемая функция f(x)
def f(x):
    return 2 * np.cos(2.5 * x) * np.exp(x / 3.0) + 4 * np.sin(3.5 * x) * np.exp(-3.0 * x) + x

# Точное значение интеграла для проверки
I =  7.077031437995793610263911711602477164432
rtol = 1e-10  # относительная точность


# 1) Точный расчет локальных весовых моментов
# Чтобы численно посчитать интеграл с особенностью, нужны локальные моменты веса
# u_k = integral_c^d (x-m)^k p(x) dx
def raw_moment_shifted(c, d, m, k):
    # Вычисляет точное значение локального момента веса:
    # integral_c^d (x - m)^k * (x - a)^(-alpha) dx
    # Использует разложение: (x - m)^k = ((x - a) + (a - m))^k
    val = 0.0
    for j in range(k + 1):
        nCr = math.comb(k, j)
        term1 = (a - m) ** (k - j)  # Сдвиг относительно левой границы a

        pow_val = j - alpha + 1.0

        upper = (d - a) ** pow_val

        # Защита от NaN при возведении нуля в степень на левой границе (когда c == a)
        lower = 0.0 if c <= a else (c - a) ** pow_val

        # Интеграл от (x - a)^(pow_val - 1) dx равен ((d - a)^p - (c - a)^p) / p
        integral_j = (upper - lower) / pow_val
        val += nCr * term1 * integral_j

    return val


# 2) Локальный трёхточечный метод Ньютона–Котса
# Это аналог метода Симпсона, адаптированный под вес
# Квадратурная формула фиксирует 3 узла:
#   1. левый край c
#   2. центр m
#   3. правый край d
#
# Формулы весов w_i получаем аналитически ч/з моменты mv_i,
# чтобы метод был абсолютно точен для любых многочленов до 2-й степени включительно
def nc3_local(c, d):
    h_loc = d - c  # Длина текущего подотрезка
    m = (c + d) / 2.0  # Средняя точка подотрезка

    # Вычисляем первые 3 момента (для степеней k = 0, 1, 2)
    mu0 = raw_moment_shifted(c, d, m, 0)
    mu1 = raw_moment_shifted(c, d, m, 1)
    mu2 = raw_moment_shifted(c, d, m, 2)

    # Нахождение весов w0, w1, w2 из условия точного интегрирования параболы
    w0 = (2.0 / h_loc ** 2) * (mu2 - (h_loc / 2.0) * mu1)
    w1 = mu0 - (4.0 / h_loc ** 2) * mu2
    w2 = (2.0 / h_loc ** 2) * (mu2 + (h_loc / 2.0) * mu1)

    # Возвращаем значение интеграла на этом подотрезке
    return w0 * f(c) + w1 * f(m) + w2 * f(d)


# 3) Локальный трёхточечный метод Гаусса-Лежандра
# В отличие от Ньютона-Котса, в методе Гаусса узлы не фиксированы
# (это не просто края и центр),
# а подбираются динамически под весовую функцию на каждом отрезке.
#
# Сначала строится ортогональный полином через матрицу Ганкеля,
# его корни дают идеальные точки (nodes)
# Затем через систему Вандермонда вычисляются веса (weights)
#
# Метод мощнее, т.е. на 3-х точках обеспечивает точность вплоть до 5-й степени многочлена
def gauss3_local(c, d):
    m = (c + d) / 2.0

    # Нам нужны 6 моментов (от 0 до 5) для построения ортогонального полинома 3-й степени
    mu = [raw_moment_shifted(c, d, m, k) for k in range(6)]

    # Составляем матрицу Ганкеля для поиска коэффициентов многочлена
    H = np.array([
        [mu[0], mu[1], mu[2]],
        [mu[1], mu[2], mu[3]],
        [mu[2], mu[3], mu[4]]
    ], dtype=float)
    B = np.array([-mu[3], -mu[4], -mu[5]], dtype=float)

    # Решаем СЛАУ H * A = B, находим коэффициенты
    A = np.linalg.solve(H, B)

    # Корни полученного многочлена — это оптимальные узлы интегрирования Гаусса
    roots_t = np.roots([1.0, A[2], A[1], A[0]])
    roots_t = np.sort(np.real(roots_t))  # Берем вещественную часть (мнимой быть не должно)

    nodes = roots_t + m  # Переводим узлы из локальных координат в глобальные

    # Составляем матрицу Вандермонда для нахождения весов квадратуры Гаусса
    V = np.array([
        [1.0, 1.0, 1.0],
        [roots_t[0], roots_t[1], roots_t[2]],
        [roots_t[0] ** 2, roots_t[1] ** 2, roots_t[2] ** 2]
    ], dtype=float)
    rhs = np.array([mu[0], mu[1], mu[2]], dtype=float)

    weights = np.linalg.solve(V, rhs)  # Решаем систему, находим веса

    # Решаем систему, находим веса
    return sum(w * f(x) for w, x in zip(weights, nodes))


# 4) Составные формулы (разбиение на N отрезков)
def composite_nc3(N):
    x = np.linspace(a, b, N + 1)  # Разбиваем отрезок [a, b] на N частей

    # Для каждого подотрезка вызываем локальное правило и суммируем результаты
    return sum(nc3_local(x[i], x[i + 1]) for i in range(N))


def composite_gauss3(N):
    x = np.linspace(a, b, N + 1)
    return sum(gauss3_local(x[i], x[i + 1]) for i in range(N))


# 5) Оценка фактического порядка сходимости по Эйткену
# Метод принимает три значения интеграла,
# посчитанных при последовательном удвоении шагов (N, 2N, 4N)
#
# Формула определяет, с какой скоростью падает погрешность:
# если p ~= 4, то при удвоении сетки ошибка падает в 2^4 = 16 раз
def aitken_p(S1, S2, S3):
    num = abs(S3 - S2)
    den = abs(S2 - S1)
    if num == 0 or den == 0:
        return np.nan
    return np.log2(den / num)  # Вычисление показателя степени сходимости p


# 6) Построение таблицы сходимости
# На каждой итерации шаг N удваивается
# Для каждой итерации считается интеграл,
# вычисляется фактическая скорость сходимости p,
# делается шаг уточнения по Ричардсону (S')
# и проверяется критерий остановки по Рунге
def build_table(method, method_name, rtol, p_theory):
    rows = []
    S_vals, Sr_vals = [], []
    N = 1

    while True:
        h = (b - a) / N  # Текущий шаг сетки
        S = method(N)  # Значение интеграла
        S_vals.append(S)

        # Заготовка под строку таблицы
        row = {
            "N": N, "h": h, "S": S,
            "E": abs(I - S) / abs(I),
            "R": np.nan, "p": np.nan,
            "S'": np.nan, "E'": np.nan,
            "p'": np.nan, "pr": np.nan
        }

        if len(S_vals) >= 2:
            if len(S_vals) >= 3:
                row["p"] = aitken_p(S_vals[-3], S_vals[-2], S_vals[-1])  # Порядок p по Эйткену

            p_used = row["p"] if not np.isnan(row["p"]) else p_theory
            S_prev = S_vals[-2]

            # Главная оценка погрешности по правилу Рунге (без знания точного ответа I)
            delta = (S - S_prev) / (2 ** p_used - 1)
            R = abs(delta) / abs(S)  # Относительная оценка Рунге
            S_rich = S + delta  # Уточнение интеграла (Экстраполяция Ричардсона)

            row["R"] = R
            row["S'"] = S_rich
            row["E'"] = abs(I - S_rich) / abs(I)  # Погрешность уточненного значения (Ричардсон)
            Sr_vals.append(S_rich)

            if len(Sr_vals) >= 3:
                row["p'"] = aitken_p(Sr_vals[-3], Sr_vals[-2], Sr_vals[-1])  # Порядок p' для Ричардсона

            # pr — локальный наклон графика погрешности в логарифмических шкалах
            if len(rows) >= 1:
                R_prev, h_prev = rows[-1]["R"], rows[-1]["h"]
                if pd.notna(R_prev) and R_prev > 0 and R > 0:
                    row["pr"] = (np.log10(R) - np.log10(R_prev)) / (np.log10(h) - np.log10(h_prev))

        rows.append(row)

        # Условие остановки: если оценка Рунге стала меньше rtol (1e-10)
        if not np.isnan(row["R"]) and row["R"] < rtol:
            break

        N *= 2  # Удваиваем число разбиений для следующей итерации

    df = pd.DataFrame(rows)
    print(f"\n{method_name}")
    pd.set_option("display.float_format", "{:.12e}".format)
    print(df.to_string(index=False))
    return df


# 7) Построение графиков
# Строит графики погрешности в масштабе log_{10}(h) до log_{10}(R)
#
# Пунктирные линии — это эталоны теории
# Если линия метода идет параллельно пунктиру наклона 4,
# значит метод имеет реальный 4-й порядок сходимости
def plot_R(df, title, slope1, slope2=None):
    d = df.dropna(subset=["R", "h"]).copy()
    h = d["h"].to_numpy()
    R = d["R"].to_numpy()

    x = np.log10(h)
    y = np.log10(R)

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, "o-", label="log10(R)")  # График погрешности

    # Опорная теоретическая прямая 1
    c1 = np.median(R / (h ** slope1))
    h_line = np.array([h.min(), h.max()])
    plt.plot(np.log10(h_line), np.log10(c1 * h_line ** slope1), "k--", label=f"опорная прямая, наклон {slope1}")

    # Опорная теоретическая прямая 2
    if slope2 is not None:
        c2 = np.median(R / (h ** slope2))
        plt.plot(np.log10(h_line), np.log10(c2 * h_line ** slope2), "g--", alpha=0.6,
                 label=f"опорная прямая, наклон {slope2}")

    plt.xlabel("log10(h)")
    plt.ylabel("log10(R)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# 8) Прогноз оптимального числа шагов N_{opt
# Позволяет не удваивать бесконечно сетку,
# а аналитически предсказать точное значение N_{opt},
# при котором погрешность впервые станет меньше rtol=10^{-10}
#
# Внутри цикла while True код проводит самоконтроль:
# запускает интегрирование с шагом N и 2N и математически проверяет,
# достигнута ли точность
def find_nopt(method, df, p_min, p_max):
    # Ищем первую строчку в таблице, где порядок p вошел в адекватные рамки
    good = df[(df["p"].notna()) & (df["p"] > p_min) & (df["p"] < p_max)].copy()
    if good.empty:
        return None

    row = good.iloc[0]
    N0 = int(row["N"])
    R0 = float(row["R"])
    p0 = float(row["p"])

    # Теоретический расчет N_est по формуле Рунге: N_opt = N0 * (R0 / rtol)^(1/p)
    N_est = int(np.ceil(N0 * (R0 / rtol) ** (1.0 / p0)))
    N = max(N_est, N0 + 1)

    while True:
        S = method(N)
        S_2N = method(N * 2)

        # Проверка по правилу Рунге для шагов N и 2N
        runge_err = abs(S_2N - S) / (2 ** p0 - 1)
        E_exact = abs(I - S_2N) / abs(I)

        # Если внутренний критерий Рунге удовлетворяет rtol, возвращаем результат
        if (runge_err / abs(S_2N)) < rtol:
            return {
                "N_start": N0,
                "p_observed": p0,
                "N_est_initial": N_est,
                "N_opt": N * 2,  # Оптимальное число шагов, гарантирующее точность
                "Runge_Relative_Error": runge_err / abs(S_2N),
                "Exact_Relative_Error": E_exact
            }
        N += 1


# Расчеты для Ньютона-Котса и Гаусса; вывод 2 таблиц; построение 2 графиков
#
# Печать в консоль найденных оптимальных значений шагов вместе с реальной ошибкой

# Запуск расчетов таблиц
nc_df = build_table(composite_nc3, "NC3 (Newton-Cotes)", rtol, p_theory=4)
gauss_df = build_table(composite_gauss3, "Gauss3 (Gauss-Legendre)", rtol, p_theory=6)

# Отрисовка графиков
plot_R(nc_df, "Newton-Cotes 3-point: log10(R) vs log10(h)", 3, 4)
plot_R(gauss_df, "Gauss 3-point: log10(R) vs log10(h)", 6)

# Поиск оптимального N по интервальным критериям (из задания)
nc_opt = find_nopt(composite_nc3, nc_df, 2.7, 4.0)
gauss_opt = find_nopt(composite_gauss3, gauss_df, 5.7, 6.7)

print("\nРезультаты подбора N_opt")
print("NC3   :", nc_opt)
print("Gauss :", gauss_opt)
