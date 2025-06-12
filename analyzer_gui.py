# analyzer_gui.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter # <-- Nowy import

# Import modeli z naszego wcześniej stworzonego pliku
try:
    import dsc_models
except ImportError:
    messagebox.showerror("Błąd importu",
                         "Nie można znaleźć pliku 'dsc_models.py'.\nUpewnij się, że ten skrypt znajduje się w tym samym folderze co 'dsc_models.py'.")
    exit()


class DSCAnalyzerApp(tk.Tk):
    """Główna klasa aplikacji do analizy danych DSC."""

    def __init__(self):
        super().__init__()
        self.title("Analizator Danych DSC")
        self.geometry("1400x800")

        # --- Inicjalizacja stanu danych ---
        self.data_state = {
            "sample_raw": None, "buffer_raw": None, "subtracted": None,
            "trimmed": None, "mhc": None, "baseline_subtracted": None,
            "fit_curve": None, "final_results": None, "baseline_params": {},
            "exclusion_range": None  # <-- DODANA LINIA
        }
        self.temp_plot_elements = {'pre': [], 'post': []}
        self.footer_text_artist = None
        self.selection_mode = None
        self.point_collector = []

        # --- Główny kontener ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Panel sterowania (lewa strona) ---
        control_frame = ttk.Frame(main_frame, width=380)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # --- Panel wykresu (prawa strona) ---
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._create_controls(control_frame)
        self._create_plot_canvas()
        self.update_button_states()

    def _create_controls(self, parent):
        """Tworzy wszystkie widżety w panelu sterowania."""
        # Kontener z przewijaniem, gdyby kontrolek było za dużo
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Sekcja 1: Wczytywanie Danych ---
        frame1 = ttk.LabelFrame(scrollable_frame, text="1. Wczytywanie Danych", padding=10)
        frame1.pack(fill=tk.X, padx=5, pady=5)

        self.btn_load_sample = ttk.Button(frame1, text="Wczytaj próbkę (CSV)", command=self.load_sample)
        self.btn_load_sample.grid(row=0, column=0, sticky=tk.EW, columnspan=2)
        self.lbl_sample_file = ttk.Label(frame1, text="Nie wczytano pliku.", style="Italic.TLabel")
        self.lbl_sample_file.grid(row=1, column=0, sticky=tk.W, columnspan=2)

        self.btn_load_buffer = ttk.Button(frame1, text="Wczytaj bufor (CSV)", command=self.load_buffer)
        self.btn_load_buffer.grid(row=2, column=0, sticky=tk.EW, columnspan=2, pady=(10, 0))
        self.lbl_buffer_file = ttk.Label(frame1, text="Nie wczytano pliku.", style="Italic.TLabel")
        self.lbl_buffer_file.grid(row=3, column=0, sticky=tk.W, columnspan=2)

        self.btn_subtract = ttk.Button(frame1, text="Odejmij bufor", command=self.subtract_buffer)
        self.btn_subtract.grid(row=4, column=0, sticky=tk.EW, columnspan=2, pady=(10, 0))

        # PRZENIESIONY PRZYCISK - Wstawiony kod
        self.btn_trim = ttk.Button(frame1, text="Ogranicz zakres temperatur", command=self.enter_trim_mode)
        self.btn_trim.grid(row=5, column=0, sticky=tk.EW, columnspan=2, pady=(10, 0))
        # KONIEC WSTAWIONEGO KODU

        # NOWY PRZYCISK - Wstawiony kod
        self.btn_reset = ttk.Button(frame1, text="Resetuj", command=self.reset_analysis)
        self.btn_reset.grid(row=6, column=0, sticky=tk.EW, columnspan=2, pady=(5, 0))
        # KONIEC WSTAWIONEGO KODU

        # --- Sekcja 2: Parametry Obliczeniowe ---
        frame2 = ttk.LabelFrame(scrollable_frame, text="2. Parametry Obliczeniowe", padding=10)
        frame2.pack(fill=tk.X, padx=5, pady=5)
        self.param_vars = {}
        params_list = {
            "Masa mol. [kDa]": ("14.307", "mass_kda"),
            "V celki [mL]": ("0.299", "vol_ml"),
            "Stężenie [mg/mL]": ("1.0", "conc_mgml"),
            "V skan. [°C/min]": ("1.0", "rate_cpmin")
        }
        for i, (text, (default, key)) in enumerate(params_list.items()):
            label = ttk.Label(frame2, text=text)
            label.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(self, value=default)
            entry = ttk.Entry(frame2, textvariable=var, width=12)
            entry.grid(row=i, column=1, sticky=tk.E, padx=5, pady=2)
            self.param_vars[key] = var

        # --- Sekcja 3: Przetwarzanie i Analiza ---
        frame3 = ttk.LabelFrame(scrollable_frame, text="3. Kroki Analizy", padding=10)
        frame3.pack(fill=tk.X, padx=5, pady=5)

        self.btn_convert = ttk.Button(frame3, text="Konwertuj na Molową Poj. Cieplną", command=self.convert_to_mhc)
        self.btn_convert.pack(fill=tk.X, pady=2)

        self.btn_base_pre = ttk.Button(frame3, text="Definiuj bazę (przed pikiem)",
                                       command=lambda: self.enter_baseline_mode('pre'))
        self.btn_base_pre.pack(fill=tk.X, pady=2)

        self.btn_base_post = ttk.Button(frame3, text="Definiuj bazę (po piku)",
                                        command=lambda: self.enter_baseline_mode('post'))
        self.btn_base_post.pack(fill=tk.X, pady=2)

        self.btn_show_base = ttk.Button(frame3, text="Pokaż linię bazową", command=self.show_baseline)
        self.btn_show_base.pack(fill=tk.X, pady=2)

        self.btn_subtract_base = ttk.Button(frame3, text="Odejmij linię bazową", command=self.subtract_baseline)
        self.btn_subtract_base.pack(fill=tk.X, pady=2)

        # --- Sekcja 4: Dopasowanie Modelu ---
        frame4 = ttk.LabelFrame(scrollable_frame, text="4. Dopasowanie Modelu", padding=10)
        frame4.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(frame4, text="Wybierz model:").pack(fill=tk.X)
        self.model_var = tk.StringVar()
        self.combo_model = ttk.Combobox(frame4, textvariable=self.model_var, state="readonly",
                                        values=['equilibrium', 'lumry-eyring', 'irreversible'])
        self.combo_model.pack(fill=tk.X, pady=2)
        self.combo_model.set('equilibrium')

        self.btn_fit = ttk.Button(frame4, text="Dopasuj model", command=self.fit_model)
        self.btn_fit.pack(fill=tk.X, pady=(5, 0))

        # --- NOWY BLOK: WYKLUCZANIE DANYCH ---
        self.btn_exclude_range = ttk.Button(frame4, text="Wyklucz/Zresetuj Zakres", command=self.enter_exclusion_mode)
        self.btn_exclude_range.pack(fill=tk.X, pady=(5, 0))
        self.lbl_exclusion_info = ttk.Label(frame4, text="Zakres wykluczony: Brak", style="Italic.TLabel")
        self.lbl_exclusion_info.pack(fill=tk.X, pady=(2, 0))
        # --- KONIEC NOWEGO BLOKU ---

        self.btn_show_residuals = ttk.Button(frame4, text="Pokaż pozostałość", command=self.show_residuals_plot)
        self.btn_show_residuals.pack(fill=tk.X, pady=(5, 0))

        # --- Sekcja 5: Zapis ---
        frame5 = ttk.LabelFrame(scrollable_frame, text="5. Zapis Wyników", padding=10)
        frame5.pack(fill=tk.X, padx=5, pady=5)
        self.btn_save = ttk.Button(frame5, text="Zapisz Raport (XLSX)", command=self.save_report)
        self.btn_save.pack(fill=tk.X)

        ttk.Style().configure("Italic.TLabel", font=("Helvetica", 9, "italic"))

    def _create_plot_canvas(self):
        """Tworzy pusty wykres Matplotlib."""
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

        self.fig.canvas.mpl_connect('button_press_event', self.on_plot_click)
        self.update_plot("Wczytaj dane, aby rozpocząć analizę")

    # --- Metody ładowania i podstawowego przetwarzania ---

    def load_file(self, target):
        filepath = filedialog.askopenfilename(
            filetypes=[("Pliki CSV", "*.csv"), ("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")])
        if not filepath: return
        try:
            # Używamy pandas do wczytania, aby obsłużyć różne formaty CSV
            # Zakładamy, że pierwsza kolumna to Temp, druga to Signal.
            # Zmieniamy header=None na header='infer' jeśli pliki mają nagłówki,
            # lub odpowiednio ustawiamy numery kolumn.
            # Ważne: Jeśli formatowanie plików różni się (np. separator, brak nagłówków),
            # to trzeba dopasować ten fragment kodu.
            df = pd.read_csv(filepath, header=None, names=['Temp', 'Signal'], comment='#', skipinitialspace=True)
            self.data_state[target] = df
            getattr(self, f"lbl_{target.split('_')[0]}_file").config(text=filepath.split('/')[-1])
            self.update_plot(f"Wczytano: {target.replace('_raw', '')}")
            self.update_button_states()
        except Exception as e:
            messagebox.showerror("Błąd odczytu pliku", f"Nie udało się wczytać pliku: {e}")

    def load_sample(self):
        self._reset_all_state_for_new_sample()  # <-- DODANA LINIA
        self.load_file("sample_raw")

    def load_buffer(self):
        self._reset_state_for_new_buffer()  # <-- DODANA LINIA
        self.load_file("buffer_raw")

    def subtract_buffer(self):
        sample = self.data_state["sample_raw"]
        buffer = self.data_state["buffer_raw"]
        if sample is None or buffer is None:
            messagebox.showerror("Błąd", "Wczytaj najpierw pliki próbki i bufora.")
            return

        interp_func = interp1d(buffer['Temp'], buffer['Signal'], bounds_error=False, fill_value="extrapolate")
        interpolated_buffer_signal = interp_func(sample['Temp'])

        subtracted_df = pd.DataFrame({
            'Temp': sample['Temp'],
            'Signal': sample['Signal'] - interpolated_buffer_signal
        })
        self.data_state['subtracted'] = subtracted_df
        self.update_plot("Po odjęciu bufora")
        self.update_button_states()

    def enter_trim_mode(self):
        current_data = self.data_state.get('subtracted')
        if current_data is None:
            messagebox.showerror("Błąd", "Najpierw odejmij bufor.")
            return

        self.selection_mode = 'trim'
        self.point_collector = []
        self.clear_temp_plot_elements()
        self.update_plot(self.ax.get_title(), "Wybierz 2 punkty, aby ograniczyć zakres temperatur")

    # --- Metody głównej analizy ---

    def convert_to_mhc(self):
        df = self.data_state['trimmed']
        if df is None: return messagebox.showerror("Błąd", "Najpierw ogranicz zakres danych.")

        try:
            params = {key: float(var.get()) for key, var in self.param_vars.items()}
        except ValueError:
            return messagebox.showerror("Błąd", "Wprowadź poprawne wartości liczbowe w parametrach.")

        # Przeliczanie stężenia mg/mL na mole
        # Zakładamy, że Masa mol. [kDa] jest w kDa, więc mnożymy przez 1000, aby uzyskać g/mol
        # Stężenie [mg/mL] dzielimy przez 1000, aby uzyskać g/mL
        # V celki [mL] dzielimy przez 1000, aby uzyskać L
        # moles = (stężenie w g/mL * objętość w L) / masa molowa w g/mol
        protein_conc_g_ml = params['conc_mgml'] / 1000.0  # g/mL
        cell_volume_L = params['vol_ml'] / 1000.0  # L
        molar_mass_g_mol = params['mass_kda'] * 1000.0  # g/mol
        moles = (protein_conc_g_ml * cell_volume_L) / molar_mass_g_mol  # moles

        rate_K_s = params['rate_cpmin'] / 60.0

        # P [µW] (zakładana jednostka surowego sygnału)
        # uW = uJ/s
        # P [uJ/s] -> Cp [J/mol·K]
        # Sygnał po odjęciu bufora i przycięciu jest w µW
        # Zakładamy, że self.data_state['trimmed']['Signal'] jest w µW
        signal_uJ_s = df['Signal']
        cp_uJ_K = signal_uJ_s / rate_K_s  # [uJ/s] / [K/s] = [uJ/K]
        mhc_uJ_molK = cp_uJ_K / moles  # [uJ/K] / [mol] = [uJ/(mol·K)]

        # Konwersja z uJ/(mol·K) na J/(mol·K)
        mhc_J_molK = mhc_uJ_molK / 1e9  # [J/(mol·K)]

        self.data_state['mhc'] = pd.DataFrame({'Temp': df['Temp'], 'MHC': mhc_J_molK})
        self.update_plot("Molowa Pojemność Cieplna (MHC)")
        self.update_button_states()

    def enter_baseline_mode(self, mode):
        """Wchodzi w tryb wyboru punktów dla linii bazowej 'pre' lub 'post'."""
        if self.data_state.get('mhc') is None:
            messagebox.showerror("Błąd", "Najpierw skonwertuj dane na MHC.")
            return

        self.selection_mode = f"baseline_{mode}"
        self.point_collector = []

        self.clear_temp_plot_elements(key=mode)

        if mode in self.data_state['baseline_params']:
            del self.data_state['baseline_params'][mode]
            self.update_button_states()

        footer_text = f"Tryb definicji bazy '{mode}': Wybierz do 3 punktów. LPM dodaje, PPM na punkcie usuwa."
        self._update_footer_text(footer_text, color="red")
        self.canvas.draw()

    def show_baseline(self):
        """Oblicza chemiczną linię bazową i wyświetla ją na wykresie z danymi MHC."""
        df = self.data_state['mhc']
        if df is None:
            return messagebox.showerror("Błąd", "Najpierw skonwertuj dane na MHC.")
        if 'pre' not in self.data_state['baseline_params'] or 'post' not in self.data_state['baseline_params']:
            return messagebox.showerror("Błąd", "Zdefiniuj obie linie bazowe (przed i po piku).")

        self.selection_mode = None
        self.clear_temp_plot_elements()

        T, Y = df['Temp'].values, df['MHC'].values

        p_pre = self.data_state['baseline_params']['pre']['poly']
        p_post = self.data_state['baseline_params']['post']['poly']

        bas1 = np.polyval(p_pre, T)
        bas2 = np.polyval(p_post, T)

        lower_boundary = np.minimum(bas1, bas2)
        upper_boundary = np.maximum(bas1, bas2)
        delta = upper_boundary - lower_boundary

        # --- KLUCZOWA POPRAWKA ZGODNIE Z PAŃSKĄ SUGESTIĄ ---
        # Aby zagwarantować, że całka będzie monotonicznie rosnąca, przesuwamy
        # całą krzywą Y w górę, tak aby jej minimum znalazło się w punkcie 0.
        # To rozwiązuje problem ujemnych wartości w danych wejściowych do całkowania.
        Y_for_integration = Y - np.min(Y)

        # Całkujemy teraz przesuniętą, zawsze nieujemną krzywą.
        Int_Y = cumulative_trapezoid(Y_for_integration, T, initial=0)

        window_length = 51
        polyorder = 3

        if len(Int_Y) < window_length:
            messagebox.showwarning("Wygładzanie",
                                   f"Zbyt mało punktów ({len(Int_Y)}) dla filtru Savitzky-Golay. Wygładzanie pominięto.")
            smoothed_int_y = Int_Y
        else:
            smoothed_int_y = savgol_filter(Int_Y, window_length, polyorder)

        min_smoothed_int_y = np.min(smoothed_int_y)
        max_smoothed_int_y = np.max(smoothed_int_y)

        if np.isclose(max_smoothed_int_y, min_smoothed_int_y):
            fraction = np.zeros_like(T)
        else:
            fraction = (smoothed_int_y - min_smoothed_int_y) / (max_smoothed_int_y - min_smoothed_int_y)
            fraction = np.clip(fraction, 0, 1)

        # Poprzednia poprawka na odwracanie 'fraction' nie jest już potrzebna,
        # ponieważ całka z 'Y_for_integration' zawsze będzie rosnąca.

        final_baseline = lower_boundary + fraction * delta

        # Wyszukiwanie Tm - znajdujemy maksymalną odległość między danymi a nową linią bazową
        Y_corrected_temp = Y - final_baseline
        tm_index = np.argmax(np.abs(Y_corrected_temp))  # Szukamy max odległości (wartość absolutna)
        Tm = T[tm_index]
        delta_cp_at_tm = upper_boundary[tm_index] - lower_boundary[tm_index]

        self.data_state['baseline_analysis'] = pd.DataFrame({'Temp': T, 'MHC': Y, 'Baseline': final_baseline})
        self.data_state['delta_cp_result'] = {'Tm': Tm, 'dCp': delta_cp_at_tm}

        self.update_plot("Wizualizacja Linii Bazowej")
        self.update_button_states()

    def subtract_baseline(self):
        """Odejmuje wcześniej obliczoną linię bazową od danych i oblicza parametry termodynamiczne."""
        df_analysis = self.data_state.get('baseline_analysis')
        if df_analysis is None:
            return messagebox.showerror("Błąd", "Najpierw oblicz i pokaż linię bazową.")

        T = df_analysis['Temp'].values
        Y = df_analysis['MHC'].values
        baseline = df_analysis['Baseline'].values
        Y_corrected = Y - baseline  # Sygnał po odjęciu bazy, w J/(mol·K)

        # Zapisujemy dane piku do dalszej analizy i wykresów
        self.data_state['baseline_subtracted'] = pd.DataFrame({'Temp': T, 'MHC_corr': Y_corrected})

        # --- NOWA CZĘŚĆ: OBLICZENIE I ZAPIS PARAMETRÓW ---

        # 1. Obliczamy entalpię van't Hoffa (pole pod pikiem)
        #    np.trapz(y, x) -> całka z Cp dT ma jednostki [J/(mol·K)] * [K] = [J/mol]
        delta_h_vh = np.trapz(Y_corrected, T)

        # 2. Pobieramy wcześniej obliczone Tm i dCp
        dcp_res = self.data_state['delta_cp_result']

        # 3. Zapisujemy wszystkie trzy parametry w jednym miejscu dla łatwego dostępu
        final_params = {
            'Tm': dcp_res['Tm'],
            'dCp': dcp_res['dCp'],  # w J/mol·K
            'dH_vH': delta_h_vh  # w J/mol
        }
        self.data_state['final_thermo_params'] = final_params
        # --- KONIEC NOWEJ CZĘŚCI ---

        self.update_plot("Pik po odjęciu linii bazowej")
        self.update_button_states()

    # Wklej tę nową metodę do klasy DSCAnalyzerApp
    def enter_exclusion_mode(self):
        """Wchodzi w tryb wyboru zakresu do wykluczenia z fitowania lub resetuje go."""
        if self.data_state.get('baseline_subtracted') is None:
            messagebox.showerror("Błąd", "Najpierw odejmij linię bazową, aby zobaczyć pik do analizy.")
            return

        # Jeśli zakres już istnieje, zapytaj o jego usunięcie
        if self.data_state['exclusion_range']:
            if messagebox.askyesno("Reset Zakresu",
                                   "Masz już zdefiniowany zakres do wykluczenia. Czy chcesz go usunąć?"):
                self.data_state['exclusion_range'] = None
                self.lbl_exclusion_info.config(text="Zakres wykluczony: Brak")
                self.update_plot("Usunięto zakres wykluczenia")
            return

        # Wejście w tryb selekcji
        self.selection_mode = 'exclude_range'
        self.point_collector = []
        self.clear_temp_plot_elements()
        self.update_plot(self.ax.get_title(), "Wybierz 2 punkty (początek i koniec) zakresu do wykluczenia")
        self._update_footer_text("Tryb wykluczania: Kliknij w dwóch miejscach, aby zdefiniować zakres.", color="red")

    def fit_model(self):
        """Dopasowuje model do piku po odjęciu linii bazowej, uwzględniając wykluczony zakres."""
        df = self.data_state.get('baseline_subtracted')
        if df is None:
            return messagebox.showerror("Błąd", "Najpierw odejmij linię bazową.")

        model_name = self.model_var.get()
        if not model_name: return messagebox.showerror("Błąd", "Wybierz model do dopasowania.")

        T_full = df['Temp'].values
        Cpex_full = df['MHC_corr'].values

        # --- NOWA LOGIKA: FILTROWANIE DANYCH PRZED FITOWANIEM ---
        exclusion_range = self.data_state.get('exclusion_range')
        if exclusion_range:
            mask = (T_full < exclusion_range[0]) | (T_full > exclusion_range[1])
            T_for_fit = T_full[mask]
            Cpex_for_fit = Cpex_full[mask]
            print(f"Fitowanie z wykluczeniem zakresu: {exclusion_range[0]:.1f}-{exclusion_range[1]:.1f} °C. "
                  f"Użyto {len(T_for_fit)} z {len(T_full)} punktów.")
        else:
            T_for_fit = T_full
            Cpex_for_fit = Cpex_full
            print("Fitowanie na pełnym zakresie danych.")
        # --- KONIEC NOWEJ LOGIKI ---

        # --- NOWY BLOK: ZAPIS DANYCH UŻYTYCH DO FITOWANIA ---
        self.data_state['data_for_fit'] = pd.DataFrame({
            'Temp': T_for_fit,
            'MHC_corr_fit': Cpex_for_fit
        })
            # --- KONIEC NOWEGO BLOKU ---
        fit_function_map = {
            'equilibrium': dsc_models.fit_equilibrium,
            'lumry-eyring': lambda t, y: dsc_models.fit_lumry_eyring(t, y, float(self.param_vars['rate_cpmin'].get())),
            'irreversible': lambda t, y: dsc_models.fit_irreversible(t, y, float(self.param_vars['rate_cpmin'].get()))
        }

        # Wywołujemy dopasowanie na potencjalnie przefiltrowanych danych
        params, metrics, _ = fit_function_map[model_name](T_for_fit, Cpex_for_fit)

        if params is None:
            messagebox.showerror("Błąd dopasowania",
                                 "Nie udało się dopasować modelu. Sprawdź parametry lub dane wejściowe.")
            return

        # --- NOWA LOGIKA: ODTWORZENIE KRZYWEJ NA PEŁNYM ZAKRESIE ---
        # Po uzyskaniu parametrów z dopasowania, generujemy krzywą teoretyczną
        # na pełnym, oryginalnym zakresie temperatur, aby móc ją porównać z wszystkimi danymi.
        rate = float(self.param_vars['rate_cpmin'].get())
        if model_name == 'equilibrium':
            fit_curve_full_range = dsc_models.model_equilibrium(T_full, *params.values())
        elif model_name == 'lumry-eyring':
            fit_curve_full_range = dsc_models.model_lumry_eyring(T_full, *params.values(), rate)
        elif model_name == 'irreversible':
            fit_curve_full_range = dsc_models.model_irreversible(T_full, *params.values(), rate)
        else:
            fit_curve_full_range = np.zeros_like(T_full)  # Fallback
        # --- KONIEC NOWEJ LOGIKI ---

        self.data_state['fit_curve'] = pd.DataFrame({'Temp': T_full, 'Fit': fit_curve_full_range})
        self.data_state['final_results'] = {'parameters': params, 'metrics': metrics}

        self.update_plot("Wynik dopasowania modelu")
        self.update_button_states()

    def save_report(self):
        if not any(value is not None for value in self.data_state.values()):
            return messagebox.showwarning("Brak danych", "Brak danych do zapisania.")

        filepath = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Plik Excel", "*.xlsx")])
        if not filepath: return

        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # --- Arkusz 1: Parametry Analizy (NOWOŚĆ) ---
                params_dict = {
                    "Parametr": [],
                    "Wartość": []
                }
                # Dodaj parametry obliczeniowe
                for key, var in self.param_vars.items():
                    params_dict["Parametr"].append(key)
                    params_dict["Wartość"].append(var.get())
                # Dodaj informacje o modelu i wykluczeniu
                if self.data_state['final_results']:
                    params_dict["Parametr"].append("Użyty model")
                    params_dict["Wartość"].append(self.model_var.get())
                if self.data_state.get('exclusion_range'):
                    er = self.data_state['exclusion_range']
                    params_dict["Parametr"].append("Zakres wykluczony [°C]")
                    params_dict["Wartość"].append(f"{er[0]:.2f} - {er[1]:.2f}")
                else:
                    params_dict["Parametr"].append("Zakres wykluczony [°C]")
                    params_dict["Wartość"].append("Brak")

                pd.DataFrame(params_dict).to_excel(writer, sheet_name='Parametry Analizy', index=False)

                # --- Arkusz 2: Wyniki Końcowe ---
                if self.data_state['final_results'] is not None:
                    params_df = pd.DataFrame.from_dict(self.data_state['final_results']['parameters'], orient='index',
                                                       columns=['Wartość'])
                    metrics_df = pd.DataFrame.from_dict(self.data_state['final_results']['metrics'], orient='index',
                                                        columns=['Wartość'])
                    report_params_df = params_df.copy()
                    if 'dH [J/mol]' in report_params_df.index:
                        report_params_df.loc['dH [kJ/mol]'] = report_params_df.loc['dH [J/mol]'] / 1000
                        report_params_df = report_params_df.drop(index='dH [J/mol]')
                    if 'Ea [J/mol]' in report_params_df.index:
                        report_params_df.loc['Ea [kJ/mol]'] = report_params_df.loc['Ea [J/mol]'] / 1000
                        report_params_df = report_params_df.drop(index='Ea [J/mol]')

                    final_summary_df = pd.concat([report_params_df, metrics_df])

                    if 'final_thermo_params' in self.data_state and self.data_state['final_thermo_params'] is not None:
                        thermo_params = self.data_state['final_thermo_params']
                        thermo_df = pd.DataFrame({
                            'Wartość': [thermo_params['Tm'], thermo_params['dCp'] / 1000, thermo_params['dH_vH'] / 1000]
                        }, index=['Tm (baseline) [°C]', 'ΔCp (baseline) [kJ/mol·K]', 'ΔH_vH (baseline) [kJ/mol]'])
                        final_summary_df = pd.concat([final_summary_df, thermo_df])

                    final_summary_df.to_excel(writer, sheet_name='Wyniki Końcowe')

                # --- Arkusze z Danymi z kolejnych etapów ---
                if self.data_state['sample_raw'] is not None:
                    self.data_state['sample_raw'].to_excel(writer, sheet_name='1_Dane_Surowe_Probka', index=False)
                if self.data_state['buffer_raw'] is not None:
                    self.data_state['buffer_raw'].to_excel(writer, sheet_name='2_Dane_Surowe_Bufor', index=False)
                if self.data_state['subtracted'] is not None:
                    self.data_state['subtracted'].to_excel(writer, sheet_name='3_Po_Odjeciu_Bufora', index=False)
                if self.data_state['trimmed'] is not None:
                    self.data_state['trimmed'].to_excel(writer, sheet_name='4_Po_Przycieciu', index=False)
                if self.data_state['mhc'] is not None:
                    self.data_state['mhc'].to_excel(writer, sheet_name='5_MHC', index=False)

                # --- Arkusz: Analiza Linii Bazowej (NOWOŚĆ) ---
                if self.data_state.get('baseline_analysis') is not None:
                    self.data_state['baseline_analysis'].to_excel(writer, sheet_name='6_Analiza_Linii_Bazowej',
                                                                  index=False)

                if self.data_state.get('baseline_subtracted') is not None:
                    self.data_state['baseline_subtracted'].to_excel(writer, sheet_name='7_Pik_Po_Odjeciu_Bazy',
                                                                    index=False)

                # --- Arkusz: Dane do Fitu (NOWOŚĆ) ---
                if self.data_state.get('data_for_fit') is not None:
                    self.data_state['data_for_fit'].to_excel(writer, sheet_name='8_Dane_Do_Fitu', index=False)

                # --- Arkusz: Dopasowanie z Residuami (ZMODYFIKOWANY) ---
                if self.data_state.get('fit_curve') is not None and self.data_state.get(
                        'baseline_subtracted') is not None:
                    df_fit = self.data_state['fit_curve'].copy()
                    df_peak = self.data_state['baseline_subtracted']
                    # Dopasowanie danych w osi X do siebie przed obliczeniem residuów
                    merged_df = pd.merge(df_peak, df_fit, on='Temp', how='left')
                    merged_df['Residuals'] = merged_df['MHC_corr'] - merged_df['Fit']
                    # Zapisujemy tylko Temp, MHC_corr, Fit i Residuals
                    merged_df[['Temp', 'MHC_corr', 'Fit', 'Residuals']].to_excel(writer,
                                                                                 sheet_name='9_Dopasowanie_i_Residua',
                                                                                 index=False)

            messagebox.showinfo("Sukces", f"Zapisano raport w pliku:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać raportu: {e}")

    # --- Metody pomocnicze i obsługi zdarzeń ---

    def on_plot_click(self, event):
        """Obsługuje kliknięcia myszą na wykresie w różnych trybach selekcji."""
        if not self.selection_mode or event.inaxes != self.ax: return

        if self.selection_mode == 'trim':
            if event.button == 1:
                self.point_collector.append((event.xdata, event.ydata))
                pt, = self.ax.plot(event.xdata, event.ydata, 'go', markersize=8, markerfacecolor='none')
                self.temp_plot_elements['pre'].append(
                    pt)  # Używamy 'pre' dla temp_plot_elements by śledzić punkty przycinania
                self.canvas.draw()
                if len(self.point_collector) == 2:
                    temps = sorted([p[0] for p in self.point_collector])
                    df_subtracted = self.data_state['subtracted']
                    idx = (df_subtracted['Temp'] >= temps[0]) & (df_subtracted['Temp'] <= temps[1])
                    self.data_state['trimmed'] = df_subtracted[idx].copy()
                    self.selection_mode = None
                    self.update_plot("Po ograniczeniu zakresu temperatur")
                    self.update_button_states()
            return

        if self.selection_mode == 'exclude_range':
            if event.button == 1:
                self.point_collector.append(event.xdata)
                pt, = self.ax.plot(event.xdata, event.ydata, 'mo', markersize=8)  # 'm' to magenta
                self.temp_plot_elements['pre'].append(pt)
                self.canvas.draw()
                if len(self.point_collector) == 2:
                    temps = sorted(self.point_collector)
                    self.data_state['exclusion_range'] = temps
                    self.selection_mode = None
                    self.lbl_exclusion_info.config(text=f"Wykluczono: {temps[0]:.1f} - {temps[1]:.1f} °C")
                    self.update_plot("Zdefiniowano zakres do wykluczenia")
                    self._update_footer_text("")
            return  # Ważne, aby zakończyć działanie metody tutaj

        if self.selection_mode in ['baseline_pre', 'baseline_post']:
            mode = self.selection_mode.split('_')[1]

            if event.button == 1 and len(self.point_collector) < 3:
                pt, = self.ax.plot(event.xdata, event.ydata, marker='o',
                                   markerfacecolor='none', markeredgecolor='red',
                                   markersize=8, linestyle='None', markeredgewidth=1.5)
                self.point_collector.append((event.xdata, event.ydata))
                self.temp_plot_elements.setdefault(mode, []).append(pt)
                self._update_temp_baseline_plot(mode)

            elif event.button == 3 and self.point_collector:
                points_to_check = self.temp_plot_elements.get(mode, [])
                markers = [el for el in points_to_check if isinstance(el, plt.Line2D) and el.get_linestyle() == 'None']

                if not markers: return

                click_pos_pixels = self.ax.transData.transform((event.xdata, event.ydata))
                point_coords = [marker.get_data() for marker in markers]
                point_pixels = [self.ax.transData.transform(np.squeeze(p)) for p in point_coords]

                distances = [np.linalg.norm(click_pos_pixels - pp) for pp in point_pixels]

                min_dist_idx = np.argmin(distances)
                if distances[min_dist_idx] < 10:
                    marker_to_remove = markers[min_dist_idx]
                    x_to_remove, y_to_remove = marker_to_remove.get_data()

                    collector_idx = -1
                    for i, (px, py) in enumerate(self.point_collector):
                        if np.isclose(px, x_to_remove) and np.isclose(py, y_to_remove):
                            collector_idx = i
                            break

                    if collector_idx != -1:
                        del self.point_collector[collector_idx]
                        marker_to_remove.remove()
                        self.temp_plot_elements[mode].remove(marker_to_remove)

                        self._update_temp_baseline_plot(mode)

    # Wklej do pliku analyzer_gui.py, zastępując istniejącą metodę update_plot

    def update_plot(self, title="", footer_text=""):
        self.clear_temp_plot_elements()
        self.ax.clear()

        current_ylabel = "Sygnał [uW]"

        if self.data_state['fit_curve'] is not None:
            df_peak = self.data_state['baseline_subtracted']
            df_fit = self.data_state['fit_curve']
            self.ax.plot(df_peak['Temp'], df_peak['MHC_corr'], 'ro', markersize=3, alpha=0.6, label="Dane")
            self.ax.plot(df_fit['Temp'], df_fit['Fit'], 'b-', linewidth=2, label="Dopasowany model")
            # --- NOWA LINIA: WIZUALIZACJA WYKLUCZONEGO ZAKRESU ---
            exclusion_range = self.data_state.get('exclusion_range')
            if exclusion_range:
                self.ax.axvspan(exclusion_range[0], exclusion_range[1], color='gray', alpha=0.3,
                                label='Zakres wykluczony')
            # --- KONIEC NOWEJ LINII ---
            self.ax.legend()
            current_ylabel = "Pojemność cieplna [J/(mol·K)]"

            # --- NOWA CZĘŚĆ: DODAWANIE POLA Z WYNIKAMI DOPASOWANIA ---
            results = self.data_state.get('final_results')
            if results:
                params = results['parameters']
                metrics = results['metrics']
                model_name = self.model_var.get().replace('-', ' ').title()

                text_lines = [f"Wyniki ({model_name}):"]

                # Pętla po parametrach i ich formatowanie
                for name, value in params.items():
                    if 'dH [J/mol]' in name:
                        text_lines.append(f"$\\Delta H$ = {value / 1000:.2f} kJ/mol")
                    elif 'Ea [J/mol]' in name:
                        text_lines.append(f"$E_a$ = {value / 1000:.2f} kJ/mol")
                    elif 'Tm [°C]' in name:
                        text_lines.append(f"$T_m$ = {value:.2f} °C")
                    elif 'T* [°C]' in name:
                        text_lines.append(f"$T^*$ = {value:.2f} °C")
                    elif 'dCp [J/mol·K]' in name:
                        # Sprawdzamy czy dCp było w ogóle fitowane
                        if 'dCp [J/mol·K]' in self.data_state['final_results']['parameters']:
                            text_lines.append(f"$\\Delta C_p$ = {value:.2f} J/mol·K")

                # Dodanie metryk dopasowania
                text_lines.append("---")
                text_lines.append(f"$R^2$ = {metrics['r_squared']:.4f}")
                text_lines.append(f"$\\chi^2_{{red}}$ = {metrics['reduced_chi_squared']:.3f}")

                text_str = "\n".join(text_lines)

                # Rysowanie pola tekstowego w lewym górnym rogu
                self.ax.text(0.05, 0.95, text_str, transform=self.ax.transAxes, fontsize=10,
                             verticalalignment='top', horizontalalignment='left',
                             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            # --- KONIEC NOWEJ CZĘŚCI ---

        elif self.data_state.get('baseline_subtracted') is not None:
            df = self.data_state['baseline_subtracted']
            self.ax.plot(df['Temp'], df['MHC_corr'], 'b-', label="Sygnał po odjęciu bazy")

            if self.data_state.get('final_thermo_params') is not None:
                params = self.data_state['final_thermo_params']
                tm_val = params['Tm']
                dcp_val_kj = params['dCp'] / 1000
                dh_val_kj = params['dH_vH'] / 1000
                text_str = (
                    f"$T_m$ = {tm_val:.2f} °C\n"
                    f"$\\Delta C_p$ = {dcp_val_kj:.2f} kJ/mol·K\n"
                    f"$\\Delta H_{{vH}}$ = {dh_val_kj:.2f} kJ/mol"
                )
                self.ax.text(0.95, 0.95, text_str, transform=self.ax.transAxes, fontsize=11,
                             verticalalignment='top', horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

            self.ax.legend(loc='lower center')
            current_ylabel = "Pojemność cieplna [J/(mol·K)]"

        elif self.data_state.get('baseline_analysis') is not None:
            df = self.data_state['baseline_analysis']
            self.ax.plot(df['Temp'], df['MHC'], label="Molowa Poj. Cieplna")
            self.ax.plot(df['Temp'], df['Baseline'], 'r--', label="Obliczona linia bazowa")
            dcp_res = self.data_state['delta_cp_result']
            text_str = f"$T_m$ = {dcp_res['Tm']:.2f} °C\n$\\Delta C_p$ = {dcp_res['dCp'] / 1000:.2f} kJ/mol·K"
            self.ax.text(0.05, 0.95, text_str, transform=self.ax.transAxes, fontsize=10,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            self.ax.legend()
            current_ylabel = "Pojemność cieplna [J/(mol·K)]"

        elif self.data_state['mhc'] is not None:
            df = self.data_state['mhc']
            self.ax.plot(df['Temp'], df['MHC'], label="Molowa Poj. Cieplna")
            self.ax.legend()
            current_ylabel = "Pojemność cieplna [J/(mol·K)]"

        elif self.data_state['trimmed'] is not None:
            df = self.data_state['trimmed']
            self.ax.plot(df['Temp'], df['Signal'], label="Sygnał po przycięciu")
            self.ax.legend()
            current_ylabel = "Sygnał [uW]"

        elif self.data_state['subtracted'] is not None:
            df = self.data_state['subtracted']
            self.ax.plot(df['Temp'], df['Signal'], label="Sygnał po odjęciu bufora")
            self.ax.legend()
            current_ylabel = "Sygnał [uW]"

        elif self.data_state['sample_raw'] is not None:
            df = self.data_state['sample_raw']
            self.ax.plot(df['Temp'], df['Signal'], label="Surowy sygnał (próbka)")
            if self.data_state['buffer_raw'] is not None:
                df_buf = self.data_state['buffer_raw']
                self.ax.plot(df_buf['Temp'], df_buf['Signal'], label="Surowy sygnał (bufor)", alpha=0.7)
            self.ax.legend()
            current_ylabel = "Sygnał [uW]"

        self.ax.set_title(title)
        self.ax.set_xlabel("Temperatura [°C]")
        self.ax.set_ylabel(current_ylabel)
        self.ax.grid(True)
        self._update_footer_text(footer_text, color="blue")
        self.fig.tight_layout(rect=[0, 0.03, 1, 1])
        self.canvas.draw()

        # NOWA METODA - Wstawiony kod
    def reset_analysis(self):
        """Resetuje stan analizy do etapu po odjęciu bufora."""
        if self.data_state.get('subtracted') is None:
            messagebox.showwarning("Brak danych", "Nie ma stanu do zresetowania.")
            return

        # Okno dialogowe z prośbą o potwierdzenie
        response = messagebox.askyesno(
            "Potwierdzenie resetu",
            "Czy na pewno chcesz zresetować analizę?\n\n"
            "Wszystkie kroki po odjęciu bufora (przycinanie, konwersja, definicja bazy, dopasowanie) zostaną utracone.",
            icon='warning'
        )

        if not response:
            return  # Użytkownik anulował reset

        # Klucze stanu do zresetowania
        keys_to_reset = [
            "trimmed", "mhc", "baseline_subtracted", "fit_curve",
            "final_results", "baseline_analysis", "delta_cp_result",
            "final_thermo_params"
        ]

        for key in keys_to_reset:
            if key in self.data_state:
                self.data_state[key] = None
        self.data_state['exclusion_range'] = None
        self.lbl_exclusion_info.config(text="Zakres wykluczony: Brak")

        # Resetowanie parametrów linii bazowej
        self.data_state['baseline_params'] = {}

        # Wyczyszczenie tymczasowych elementów z wykresu (np. punktów bazy)
        self.clear_temp_plot_elements()

        # Zaktualizowanie interfejsu
        self.update_plot("Zresetowano do etapu po odjęciu bufora")
        self.update_button_states()

        messagebox.showinfo("Sukces", "Analiza została zresetowana.")
    # KONIEC WSTAWIONEGO KODU

        # NOWA METODA - Wstawiony kod
    def _reset_all_state_for_new_sample(self):
        """
        Całkowicie resetuje stan aplikacji. Czyści wszystkie dane,
        analizy i resetuje etykiety plików.
        """
        # Przywrócenie słownika data_state do stanu początkowego
        # --- Inicjalizacja stanu danych ---
        self.data_state = {
            "sample_raw": None, "buffer_raw": None, "subtracted": None,
            "trimmed": None, "mhc": None, "baseline_subtracted": None,
            # Zmieniono 'peak_only' na 'baseline_subtracted' dla spójności
            "fit_curve": None, "final_results": None, "baseline_params": {},
            "exclusion_range": None  # <-- KLUCZOWA DODANA LINIA
        }
        # Resetowanie dodatkowych stanów, jeśli istnieją
        if 'baseline_analysis' in self.data_state:
            self.data_state['baseline_analysis'] = None
        if 'delta_cp_result' in self.data_state:
            self.data_state['delta_cp_result'] = None
        if 'final_thermo_params' in self.data_state:
            self.data_state['final_thermo_params'] = None

        # Zresetowanie etykiet plików w GUI
        self.lbl_sample_file.config(text="Nie wczytano pliku.")
        self.lbl_buffer_file.config(text="Nie wczytano pliku.")

        # Wyczyszczenie tymczasowych elementów z wykresu
        self.clear_temp_plot_elements()

        # Zresetowanie trybu selekcji
        self.selection_mode = None

        # NOWA METODA - Wstawiony kod
    def _reset_state_for_new_buffer(self):
        """
        Resetuje stan analizy, zachowując dane próbki, ale czyszcząc
        stary bufor i wszystkie kolejne kroki analizy.
        """
        # Klucze do zresetowania - nie ruszamy 'sample_raw'
        keys_to_reset = [
            "buffer_raw", "subtracted", "trimmed", "mhc",
            "baseline_subtracted", "fit_curve", "final_results",
            "baseline_analysis", "delta_cp_result", "final_thermo_params"
        ]

        for key in keys_to_reset:
            if key in self.data_state:
                self.data_state[key] = None

        # Resetowanie parametrów linii bazowej
        self.data_state['baseline_params'] = {}

        # Zresetowanie tylko etykiety pliku bufora
        self.lbl_buffer_file.config(text="Nie wczytano pliku.")

        # Wyczyszczenie tymczasowych elementów z wykresu
        self.clear_temp_plot_elements()

        # Zresetowanie trybu selekcji
        self.selection_mode = None

    def clear_temp_plot_elements(self, key=None):
        """Czyści tymczasowe elementy z wykresu."""
        if key:
            for element in self.temp_plot_elements.get(key, []):
                element.remove()
            self.temp_plot_elements[key] = []
        else:
            for key_ in self.temp_plot_elements:
                for element in self.temp_plot_elements[key_]:
                    element.remove()
                self.temp_plot_elements[key_] = []

        self.canvas.draw() if hasattr(self, 'canvas') else None

    def _update_temp_baseline_plot(self, mode):
        """Aktualizuje tymczasowy wykres linii bazowej (liniowej dla 2 pkt, kwadratowej dla 3)."""
        elements_in_mode = self.temp_plot_elements.get(mode, [])
        old_lines = [el for el in elements_in_mode if isinstance(el, plt.Line2D) and el.get_linestyle() != 'None']
        for line in old_lines:
            line.remove()
            self.temp_plot_elements[mode].remove(line)

        points = np.array(self.point_collector)
        if len(points) < 2:
            self.canvas.draw()
            return

        points = points[points[:, 0].argsort()]
        poly_deg = 1 if len(points) == 2 else 2
        p = np.polyfit(points[:, 0], points[:, 1], poly_deg)

        x_fit = np.linspace(points[0, 0], points[-1, 0], 100)
        y_fit = np.polyval(p, x_fit)
        line, = self.ax.plot(x_fit, y_fit, 'r--')
        self.temp_plot_elements[mode].append(line)

        if len(points) >= 2:  # Zmieniono z ==3 na >=2, aby aktywować przyciski po wybraniu min. 2 punktów
            self.data_state['baseline_params'][mode] = {'points': points, 'poly': p}
            self.update_button_states()

        self.canvas.draw()

    def _update_footer_text(self, text="", color="blue"):
        """Usuwa stary tekst stopki i rysuje nowy, zarządzając jego obiektem."""
        if self.footer_text_artist:
            self.footer_text_artist.remove()

        if text:
            bbox_props = dict(boxstyle='round', facecolor='white', alpha=0.8,
                              edgecolor=color) if color == 'red' else None
            self.footer_text_artist = self.fig.text(
                0.5, 0.01, text, ha='center', color=color, fontsize=9, bbox=bbox_props
            )
        else:
            self.footer_text_artist = None

        self.canvas.draw_idle()

    def update_button_states(self):
        """Włącza/wyłącza przyciski w zależności od stanu analizy."""
        s = self.data_state

        can_subtract_buffer = s['sample_raw'] is not None and s['buffer_raw'] is not None
        can_reset = s['trimmed'] is not None  # NOWA LINIA
        can_trim = s['subtracted'] is not None
        can_convert = s['trimmed'] is not None
        can_define_base = s['mhc'] is not None
        can_show_base = 'pre' in s['baseline_params'] and 'post' in s['baseline_params']
        can_subtract_base = s.get('baseline_analysis') is not None
        can_fit = s.get('baseline_subtracted') is not None  # Zmieniono z 'peak_only' na 'baseline_subtracted'
        can_show_residuals = s.get('fit_curve') is not None  # NOWA LINIA
        can_save = any(value is not None for value in s.values())

        self.btn_subtract['state'] = 'normal' if can_subtract_buffer else 'disabled'
        self.btn_trim['state'] = 'normal' if can_trim else 'disabled'
        self.btn_reset['state'] = 'normal' if can_reset else 'disabled'  # NOWA LINIA
        self.btn_convert['state'] = 'normal' if can_convert else 'disabled'
        self.btn_base_pre['state'] = 'normal' if can_define_base else 'disabled'
        self.btn_base_post['state'] = 'normal' if can_define_base else 'disabled'
        self.btn_show_base['state'] = 'normal' if can_show_base else 'disabled'
        self.btn_subtract_base['state'] = 'normal' if can_subtract_base else 'disabled'
        self.btn_fit['state'] = 'normal' if can_fit else 'disabled'
        self.btn_save['state'] = 'normal' if can_save else 'disabled'
        # NOWA LINIA - Wstawiony kod
        self.btn_show_residuals['state'] = 'normal' if can_show_residuals else 'disabled'
        # KONIEC WSTAWIONEGO KODU

# NOWA METODA - Wstawiony kod
    def show_residuals_plot(self):
        """Tworzy nowe okno i rysuje w nim wykres pozostałości."""
        df_peak = self.data_state.get('baseline_subtracted')
        df_fit = self.data_state.get('fit_curve')

        if df_peak is None or df_fit is None:
            messagebox.showwarning("Brak danych", "Aby pokazać pozostałości, najpierw dopasuj model.")
            return

        # Utworzenie nowego okna (Toplevel)
        residual_window = tk.Toplevel(self)
        residual_window.title("Wykres Pozostałości")
        residual_window.geometry("800x600")

        # Utworzenie figury i osi dla wykresu
        fig, ax = plt.subplots(figsize=(7, 5))

        # Obliczenie pozostałości
        temperature = df_peak['Temp'].values
        residuals = df_peak['MHC_corr'].values - df_fit['Fit'].values

        # Rysowanie danych
        ax.plot(temperature, residuals, 'o-', markersize=3, label="Pozostałości (dane - model)")
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label="Linia zerowa")

        # Ustawienia wykresu
        ax.set_title("Pozostałości po dopasowaniu modelu")
        ax.set_xlabel("Temperatura [°C]")
        ax.set_ylabel("Różnica [J/(mol·K)]")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        # Osadzenie wykresu w oknie Tkinter
        canvas = FigureCanvasTkAgg(fig, master=residual_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Dodanie paska narzędzi Matplotlib
        toolbar = NavigationToolbar2Tk(canvas, residual_window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    # KONIEC WSTAWIONEGO KODU


if __name__ == "__main__":
    app = DSCAnalyzerApp()
    app.mainloop()