# dsc_models.py

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid

# Stała gazowa w jednostkach SI (J/mol·K)
R_SI = 8.314  # J/mol·K


def evaluate_fit(y_observed, y_fitted, num_params):
    """
    Oblicza zestaw metryk do oceny jakości dopasowania.

    Argumenty:
        y_observed (np.array): Dane eksperymentalne.
        y_fitted (np.array): Dane z dopasowanego modelu.
        num_params (int): Liczba parametrów w modelu.

    Zwraca:
        dict: Słownik zawierający obliczone metryki.
    """
    residuals = y_observed - y_fitted
    n_points = len(y_observed)

    # Metryka 1: Współczynnik determinacji (R-squared)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_observed - np.mean(y_observed)) ** 2)
    if ss_tot == 0:
        r_squared = 1.0 if ss_res == 0 else 0.0
    else:
        r_squared = 1 - (ss_res / ss_tot)

    # Metryka 2: Zredukowany Chi-kwadrat (najbardziej zalecana metoda)
    # Estymacja wariancji szumu na podstawie pierwszych/ostatnich 10% punktów
    baseline_points = int(n_points * 0.1)
    if baseline_points > 1:
        noise_variance = np.var(np.concatenate([y_observed[:baseline_points], y_observed[-baseline_points:]]), ddof=1)
    else:  # Fallback dla bardzo małej liczby punktów
        noise_variance = np.var(y_observed, ddof=1)

    if noise_variance == 0:
        reduced_chi2 = np.inf
    else:
        degrees_of_freedom = n_points - num_params
        if degrees_of_freedom <= 0:
            reduced_chi2 = np.inf  # Nie można obliczyć
        else:
            chi2 = np.sum(residuals ** 2 / noise_variance)
            reduced_chi2 = chi2 / degrees_of_freedom

    return {
        'r_squared': r_squared,
        'reduced_chi_squared': reduced_chi2,
    }


# --- Model 1: Równowagowy (fitdsc1.m) ---

def model_equilibrium(T_C, dH, Tm):
    """
    Model równowagowy dwustanowy (N <=> U).
    Jednostki parametrów: dH [J/mol], Tm [°C], dCp [J/mol·K]
    """
    T_K = T_C + 273.15
    Tm_K = Tm + 273.15
    K = np.exp((dH / R_SI) * (1 / Tm_K - 1 / T_K))
    Cpex = dH ** 2 / (R_SI * T_K ** 2) * (K / (1 + K) ** 2)

    return Cpex


def fit_equilibrium(T, Cpex_data):
    """Dopasowuje dane do modelu równowagowego."""
    print("Rozpoczynanie dopasowania do modelu równowagowego...")
    initial_guess = [500000, 65]
    bounds = ([0, 0], [1000000, 120])
    param_names = ['dH [J/mol]', 'Tm [°C]']

    try:
        params, _ = curve_fit(model_equilibrium, T, Cpex_data, p0=initial_guess, bounds=bounds, maxfev=50000)
        Cpex_fit = model_equilibrium(T, *params)

        metrics = evaluate_fit(Cpex_data, Cpex_fit, num_params=len(params))
        param_dict = dict(zip(param_names, params))

        print("Dopasowanie zakończone sukcesem.")
        return param_dict, metrics, Cpex_fit

    except RuntimeError as e:
        print(f"Dopasowanie do modelu równowagowego nie powiodło się: {e}")
        return None, None, None


# --- Model 2: Lumry-Eyring (fitdsc2.m) ---

def model_lumry_eyring(T_C, dH, Tm, Ea, Ta, heating_rate_C_min):
    """
    Model kinetyczny Lumry-Eyring (N <=> U -> D).
    Jednostki: dH, Ea [J/mol], Tm, Ta [°C]
    """
    T_K = T_C + 273.15
    Tm_K = Tm + 273.15
    Ta_K = Ta + 273.15
    v_K_s = heating_rate_C_min #/ 60  # Szybkość grzania w K/s

    K = np.exp((-dH / R_SI) * (1 / T_K - 1 / Tm_K))
    k = np.exp((-Ea / R_SI) * (1 / T_K - 1 / Ta_K))

    KkK = (K * k) / (K + 1)

    integral_term = cumulative_trapezoid(KkK, T_K, initial=0)

    Cpex = ((K * dH) / ((K + 1) ** 2)) * ((k / v_K_s) + dH / (R_SI * T_K ** 2)) * np.exp(
        (-1 / v_K_s) * integral_term)
    return Cpex


def fit_lumry_eyring(T, Cpex_data, heating_rate_C_min):
    """Dopasowuje dane do modelu Lumry-Eyring."""
    print("Rozpoczynanie dopasowania do modelu Lumry-Eyring...")
    model_func = lambda T_C, dH, Tm, Ea, Ta: model_lumry_eyring(T_C, dH, Tm, Ea, Ta, heating_rate_C_min)

    initial_guess = [500000, 65, 500000, 75]
    bounds = ([0, 0, 0, 0], [2000000, 120, 2000000, 120])
    param_names = ['dH [J/mol]', 'Tm [°C]', 'Ea [J/mol]', 'T* [°C]']

    try:
        params, _ = curve_fit(model_func, T, Cpex_data, p0=initial_guess, bounds=bounds, maxfev=5000)
        Cpex_fit = model_func(T, *params)

        metrics = evaluate_fit(Cpex_data, Cpex_fit, num_params=len(params))
        param_dict = dict(zip(param_names, params))

        print("Dopasowanie zakończone sukcesem.")
        return param_dict, metrics, Cpex_fit

    except RuntimeError as e:
        print(f"Dopasowanie do modelu Lumry-Eyring nie powiodło się: {e}")
        return None, None, None


# --- Model 3: Nieodwracalny N->D (fitdsc3.m) ---

def model_irreversible(T_C, dH, Ea, Ta, heating_rate_C_min):
    """
    Model kinetyczny nieodwracalny (N -> D).
    Jednostki: dH, Ea [J/mol], Ta [°C]
    """
    T_K = T_C + 273.15
    Ta_K = Ta + 273.15
    v_K_s = heating_rate_C_min / 60  # K/s

    k = np.exp((-Ea / R_SI) * (1 / T_K - 1 / Ta_K))

    integral_term = cumulative_trapezoid(k, T_K, initial=0)

    Cpex = (k / v_K_s) * dH * np.exp((-1 / v_K_s) * integral_term)
    return Cpex


def fit_irreversible(T, Cpex_data, heating_rate_C_min):
    """Dopasowuje dane do modelu nieodwracalnego N->D."""
    print("Rozpoczynanie dopasowania do modelu nieodwracalnego (N->D)...")
    model_func = lambda T_C, dH, Ea, Ta: model_irreversible(T_C, dH, Ea, Ta, heating_rate_C_min)

    initial_guess = [500000, 500000, 65]
    bounds = ([0, 0, 0], [2000000, 2000000, 150])
    param_names = ['dH [J/mol]', 'Ea [J/mol]', 'T* [°C]']

    try:
        params, _ = curve_fit(model_func, T, Cpex_data, p0=initial_guess, bounds=bounds, maxfev=5000)
        Cpex_fit = model_func(T, *params)

        metrics = evaluate_fit(Cpex_data, Cpex_fit, num_params=len(params))
        param_dict = dict(zip(param_names, params))

        print("Dopasowanie zakończone sukcesem.")
        return param_dict, metrics, Cpex_fit

    except RuntimeError as e:
        print(f"Dopasowanie do modelu nieodwracalnego nie powiodło się: {e}")
        return None, None, None