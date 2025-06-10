# example.py

import numpy as np
from dsc_experiment import DSCExperiment


def generate_mock_data(temp_range=(20, 100), n_points=1000, tm=65, dh=120, baseline_slope=0.01):
    """Generuje realistycznie wyglądające dane DSC."""
    T = np.linspace(temp_range[0], temp_range[1], n_points)
    # Prosty model Gaussa jako pik
    peak = dh * np.exp(-((T - tm) ** 2) / 10)
    # Liniowa linia bazowa z szumem
    baseline = baseline_slope * T + np.random.normal(0, 0.5, n_points)
    return T, peak + baseline


def main():
    """Główny skrypt demonstrujący przepływ pracy."""

    # --- Krok 1: Przygotowanie danych (tutaj generujemy sztuczne dane) ---
    print("Generowanie sztucznych danych...")
    # Dane dla próbki białka
    temp_sample, signal_sample = generate_mock_data(tm=70, dh=150, baseline_slope=0.2)
    protein_scan = DSCExperiment(temp_sample, signal_sample, name="Próbka Białka")

    # Dane dla bufora (zwykle bardziej płaskie)
    temp_buffer, signal_buffer = generate_mock_data(tm=70, dh=0, baseline_slope=0.04)
    buffer_scan = DSCExperiment(temp_buffer, signal_buffer, name="Bufor")

    protein_scan.plot('raw')

    # --- Krok 2: Przetwarzanie danych ---
    # W prawdziwym użyciu:
    # protein_scan = DSCExperiment.load_from_file("sciezka/do/probki.txt")
    # buffer_scan = DSCExperiment.load_from_file("sciezka/do/bufora.txt")

    # Parametry dla przetwarzania
    MIN_TEMP = 30
    MAX_TEMP = 95
    PROTEIN_CONC_MM = 0.5  # mM
    HEATING_RATE = 60  # °C/min

    protein_scan.process_scan(
        buffer_exp=buffer_scan,
        min_temp=MIN_TEMP,
        max_temp=MAX_TEMP,
        protein_conc_mM=PROTEIN_CONC_MM,
        heating_rate_C_min=HEATING_RATE
    )
    protein_scan.plot('processed')

    # --- Krok 3 (Opcjonalny): Analiza Van't Hoffa ---
    # protein_scan.analyze_baseline_vanthoff(pre_trans_range=(35, 45), post_trans_range=(85, 95))

    # --- Krok 4: Dopasowanie do modelu ---
    # Wybierz model: 'equilibrium', 'lumry-eyring', 'irreversible'
    protein_scan.fit(model_type='equilibrium')

    # --- Krok 5: Wyświetl wszystkie zebrane wyniki ---
    protein_scan.show_results()


if __name__ == "__main__":
    main()