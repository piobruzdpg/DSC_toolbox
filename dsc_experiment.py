# dsc_experiment.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import dsc_models as models


class DSCExperiment:
    """
    Klasa do przechowywania i analizy danych z eksperymentu DSC.
    """

    def __init__(self, temp_data, signal_data, name="Eksperyment"):
        self.name = name
        self.temp_raw = np.array(temp_data)
        self.signal_raw = np.array(signal_data)

        self.temp_processed = None
        self.signal_processed = None

        self.results = {}
        self.fit_curve = None
        print(f"Utworzono obiekt '{self.name}' z {len(self.temp_raw)} punktami danych.")

    @staticmethod
    def load_from_file(filepath, **kwargs):
        """Metoda statyczna do wczytywania danych z pliku tekstowego."""
        try:
            data = np.loadtxt(filepath, **kwargs)
            temp = data[:, 0]
            signal = data[:, 1]
            return DSCExperiment(temp, signal, name=filepath)
        except Exception as e:
            print(f"Błąd podczas wczytywania pliku {filepath}: {e}")
            return None

    def downsample(self, num_points=500):
        """
        Zmniejsza liczbę punktów danych (odpowiednik shrink.m).
        """
        if self.temp_processed is None:
            target_temp, target_signal = self.temp_raw, self.signal_raw
        else:
            target_temp, target_signal = self.temp_processed, self.signal_processed

        if len(target_temp) <= num_points:
            print("Dane już mają wystarczająco mało punktów. Pomijanie.")
            return

        step = round(len(target_temp) / num_points)
        self.temp_processed = target_temp[::step]
        self.signal_processed = target_signal[::step]
        print(f"Zmniejszono liczbę punktów z {len(target_temp)} do {len(self.temp_processed)}.")

    def process_scan(self, buffer_exp, min_temp, max_temp, protein_conc_mM, heating_rate_C_min):
        """
        Przetwarza surowe dane: odejmuje bufor, przycina i konwertuje jednostki.
        Odpowiednik wyrownanie.m i convert.m.
        """
        print("Rozpoczynanie przetwarzania skanu...")
        interp_func = interp1d(buffer_exp.temp_raw, buffer_exp.signal_raw, bounds_error=False, fill_value="extrapolate")
        buf_interp = interp_func(self.temp_raw)
        signal_corrected = self.signal_raw - buf_interp

        indices = np.where((self.temp_raw >= min_temp) & (self.temp_raw <= max_temp))
        temp_trimmed = self.temp_raw[indices]
        signal_trimmed = signal_corrected[indices]
        signal_trimmed -= np.min(signal_trimmed)

        cell_volume_ml = 0.299
        moles = (protein_conc_mM / 1000) * (cell_volume_ml / 1000)
        heating_rate_K_s = heating_rate_C_min / 60

        signal_uJ_K = signal_trimmed / heating_rate_K_s
        signal_uJ_Kmol = signal_uJ_K / moles
        signal_kJ_Kmol = signal_uJ_Kmol / 1e9

        self.temp_processed = temp_trimmed
        self.signal_processed = signal_kJ_Kmol
        self.results['concentration_mM'] = protein_conc_mM
        self.results['heating_rate_C_min'] = heating_rate_C_min
        print("Przetwarzanie skanu zakończone.")

    def analyze_baseline_vanthoff(self, pre_trans_range, post_trans_range):
        """
        Wyznacza linię bazową i oblicza entalpię kalorymetryczną (linbasevh.m).
        UWAGA: Ta metoda operuje na już przetworzonych danych.
        """
        if self.temp_processed is None:
            print("Błąd: Dane muszą być najpierw przetworzone metodą process_scan().")
            return

        T = self.temp_processed
        Y = self.signal_processed

        p1, p2 = np.searchsorted(T, pre_trans_range)
        b1, b2 = np.searchsorted(T, post_trans_range)

        P1 = np.polyfit(T[p1:p2], Y[p1:p2], 1)
        bas1 = np.polyval(P1, T)
        P2 = np.polyfit(T[b1:b2], Y[b1:b2], 1)
        bas2 = np.polyval(P2, T)

        Int_Y = cumulative_trapezoid(Y, T, initial=0)
        I1 = np.polyfit(T[p1:p2], Int_Y[p1:p2], 1)
        Ib1 = np.polyval(I1, T)
        I2 = np.polyfit(T[b1:b2], Int_Y[b1:b2], 1)
        Ib2 = np.polyval(I2, T)

        Bas = (Int_Y - Ib1) / (Ib2 - Ib1)
        Bas = np.clip(Bas, 0, 1)

        sigmoidal_baseline = bas1 + Bas * (bas2 - bas1)

        Y_corrected = Y - sigmoidal_baseline
        delta_H_cal = np.trapz(Y_corrected, T)

        half_point_idx = np.searchsorted(Bas, 0.5)
        dCp = np.polyval(P2, T[half_point_idx]) - np.polyval(P1, T[half_point_idx])

        self.signal_processed = Y_corrected
        self.results['delta_H_calorimetric'] = delta_H_cal
        self.results['delta_Cp'] = dCp
        print("Analiza linii bazowej i entalpii kalorymetrycznej zakończona.")
        self.plot(plot_type='baseline_analysis',
                  baselines={'pre': bas1, 'post': bas2, 'sigmoidal': sigmoidal_baseline, 'original_signal': Y})

    def fit(self, model_type='equilibrium'):
        """Dopasowuje przetworzone dane do wybranego modelu."""
        if self.temp_processed is None:
            print("Błąd: Dane muszą być najpierw przetworzone.")
            return

        T = self.temp_processed
        Cpex = self.signal_processed

        fit_function_map = {
            'equilibrium': models.fit_equilibrium,
            'lumry-eyring': lambda t, y: models.fit_lumry_eyring(t, y, self.results.get('heating_rate_C_min')),
            'irreversible': lambda t, y: models.fit_irreversible(t, y, self.results.get('heating_rate_C_min'))
        }

        if model_type not in fit_function_map:
            print(f"Błąd: Nieznany typ modelu '{model_type}'. Dostępne: {list(fit_function_map.keys())}")
            return

        if model_type in ['lumry-eyring', 'irreversible'] and 'heating_rate_C_min' not in self.results:
            print(f"Błąd: Model '{model_type}' wymaga szybkości grzania. Uruchom najpierw process_scan().")
            return

        params, metrics, fit_curve = fit_function_map[model_type](T, Cpex)

        if params and metrics:
            self.results[f'fit_{model_type}'] = {
                'parameters': params,
                'metrics': metrics
            }
            self.fit_curve = fit_curve
            self.plot(plot_type='fit', model_name=model_type)
            self.show_results(model_type=f'fit_{model_type}')

    def show_results(self, model_type=None):
        """Wyświetla podsumowanie wyników."""
        print("\n" + "=" * 40)
        print("--- PODSUMOWANIE WYNIKÓW ---")
        print("=" * 40)
        if not self.results:
            print("Brak wyników do wyświetlenia.")
            return

        for key, value in self.results.items():
            if not key.startswith('fit_') and not key.startswith('raw_'):
                print(f"{key:<25}: {value}")

        for key, value in self.results.items():
            if key.startswith('fit_') and (model_type is None or key == model_type):
                print(f"\n--- Wyniki dla modelu: '{key.replace('fit_', '')}' ---")

                if 'parameters' in value:
                    print("  Dopasowane parametry:")
                    for param, p_val in value['parameters'].items():
                        print(f"    {param:<10}: {p_val:.4f}")

                if 'metrics' in value:
                    print("  Metryki jakości dopasowania:")
                    for metric, m_val in value['metrics'].items():
                        print(f"    {metric:<20}: {m_val:.5f}")
        print("\n" + "=" * 40)

    def plot(self, plot_type='processed', **kwargs):
        """Wizualizuje dane na różnych etapach analizy."""
        plt.figure(figsize=(10, 7))

        if plot_type == 'raw':
            plt.plot(self.temp_raw, self.signal_raw, 'k-', label='Dane surowe')
            plt.title(f'Surowe dane dla {self.name}')

        elif plot_type == 'processed':
            if self.signal_processed is None:
                print("Brak przetworzonych danych do wyświetlenia.")
                return
            plt.plot(self.temp_processed, self.signal_processed, 'b-', label='Dane przetworzone')
            plt.title(f'Przetworzone dane dla {self.name}')

        elif plot_type == 'fit':
            model_name = kwargs.get('model_name', 'model')
            if self.signal_processed is None or self.fit_curve is None:
                print("Brak danych lub dopasowania do wyświetlenia.")
                return
            plt.plot(self.temp_processed, self.signal_processed, 'ro', label='Dane przetworzone', markersize=4,
                     alpha=0.6)
            plt.plot(self.temp_processed, self.fit_curve, 'b-', label=f'Dopasowanie ({model_name})', linewidth=2)
            plt.title(f'Wynik dopasowania do modelu: {model_name}')

        elif plot_type == 'baseline_analysis':
            baselines = kwargs.get('baselines', {})
            plt.plot(self.temp_processed, baselines.get('original_signal'), 'k-',
                     label='Oryginalny sygnał (po odjęciu bufora)')
            if 'pre' in baselines:
                plt.plot(self.temp_processed, baselines['pre'], 'r--', label='Linia bazowa (pre)')
            if 'post' in baselines:
                plt.plot(self.temp_processed, baselines['post'], 'g--', label='Linia bazowa (post)')
            if 'sigmoidal' in baselines:
                plt.plot(self.temp_processed, baselines['sigmoidal'], 'b-', label='Linia bazowa (sigmoidalna)',
                         linewidth=2)
            plt.title('Analiza i korekcja linii bazowej')

        else:
            print(f"Nieznany typ wykresu: {plot_type}")

        plt.xlabel('Temperatura [°C]')
        plt.ylabel('Sygnał [zależny od etapu]')
        plt.grid(True)
        plt.legend()
        plt.show()