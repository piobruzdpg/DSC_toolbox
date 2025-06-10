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
            "fit_curve": None, "final_results": None, "baseline_params": {}
        }
        self.temp_plot_elements = {'pre': [], 'post': []}
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

        # --- Sekcja 2: Parametry Obliczeniowe ---
        frame2 = ttk.LabelFrame(scrollable_frame, text="2. Parametry Obliczeniowe", padding=10)
        frame2.pack(fill=tk.X, padx=5, pady=5)
        self.param_vars = {}
        params_list = {
            "Masa mol. [kDa]": ("14.0", "mass_kda"),
            "V celki [mL]": ("0.299", "vol_ml"),
            "Stężenie [mg/mL]": ("1.0", "conc_mgml"),
            "V skan. [°C/min]": ("60.0", "rate_cpmin")
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

        self.btn_trim = ttk.Button(frame3, text="Ogranicz zakres temperatur", command=self.enter_trim_mode)
        self.btn_trim.pack(fill=tk.X, pady=2)

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
            df = pd.read_csv(filepath, header=None, names=['Temp', 'Signal'], comment='#', skipinitialspace=True)
            self.data_state[target] = df
            getattr(self, f"lbl_{target.split('_')[0]}_file").config(text=filepath.split('/')[-1])
            self.update_plot(f"Wczytano: {target}")
            self.update_button_states()
        except Exception as e:
            messagebox.showerror("Błąd odczytu pliku", f"Nie udało się wczytać pliku: {e}")

    def load_sample(self):
        self.load_file("sample_raw")

    def load_buffer(self):
        self.load_file("buffer_raw")

    def subtract_buffer(self):
        sample = self.data_state["sample_raw"]
        buffer = self.data_state["buffer_raw"]
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

        mass_g = (params['conc_mgml'] * params['vol_ml']) / 1000.0
        molar_mass_g_mol = params['mass_kda'] * 1000.0
        moles = mass_g / molar_mass_g_mol

        rate_K_s = params['rate_cpmin'] / 60.0

        # P [µJ/s] -> Cp [J/mol·K]
        signal_J_s = df['Signal'] / 1_000_000.0
        cp_J_K = signal_J_s / rate_K_s
        mhc_J_molK = cp_J_K / moles

        self.data_state['mhc'] = pd.DataFrame({'Temp': df['Temp'], 'MHC': mhc_J_molK})
        self.update_plot("Molowa Pojemność Cieplna (MHC)")
        self.update_button_states()

    # Wklej w miejsce starej metody enter_baseline_mode
    def enter_baseline_mode(self, mode):
        """Wchodzi w tryb wyboru punktów dla linii bazowej 'pre' lub 'post'."""
        self.selection_mode = f"baseline_{mode}"
        self.point_collector = []

        # ZMIANA: Usuń poprzednie punkty i linię tylko dla TEGO trybu ('pre' lub 'post')
        self.clear_temp_plot_elements(key=mode)

        # ZMIANA: Usuwamy parametry tylko wtedy, gdy zaczynamy definiować daną bazę od nowa.
        if mode in self.data_state['baseline_params']:
            del self.data_state['baseline_params'][mode]
            # Po usunięciu parametrów, musimy zaktualizować stan przycisków
            self.update_button_states()

        # ZMIANA: Zamiast przerysowywać cały wykres (co wszystko kasuje),
        # wyświetlamy instrukcję w stopce.
        footer_text = f"Tryb definicji bazy '{mode}': Wybierz do 3 punktów. LPM dodaje, PPM na punkcie usuwa."
        self.fig.text(0.5, 0.01, footer_text, ha='center', color='red', fontsize=10,
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
        self.canvas.draw()

    def show_baseline(self):
        """Oblicza chemiczną linię bazową i wyświetla ją na wykresie z danymi MHC."""
        # ... (cała logika obliczeniowa pozostaje bez zmian) ...
        df = self.data_state['mhc']
        if df is None: return messagebox.showerror("Błąd", "Najpierw skonwertuj dane na MHC.")
        if 'pre' not in self.data_state['baseline_params'] or 'post' not in self.data_state['baseline_params']:
            return messagebox.showerror("Błąd", "Zdefiniuj obie linie bazowe (przed i po piku).")

        # --- Nowy początek funkcji: wyczyść tymczasowe elementy ---
        self.selection_mode = None  # Zakończ tryb selekcji
        self.clear_temp_plot_elements()  # Wyczyść WSZYSTKIE punkty i linie pomocnicze

        T, Y = df['Temp'].values, df['MHC'].values
        p_pre = self.data_state['baseline_params']['pre']['poly']
        p_post = self.data_state['baseline_params']['post']['poly']

        bas1 = np.polyval(p_pre, T)
        bas2 = np.polyval(p_post, T)

        # ... (reszta funkcji bez zmian) ...
        Int_Y = cumulative_trapezoid(Y, T, initial=0)
        idx_pre = np.where(T < self.data_state['baseline_params']['pre']['points'][0, 0])[0]
        idx_post = np.where(T > self.data_state['baseline_params']['post']['points'][-1, 0])[0]
        if len(idx_pre) < 2 or len(idx_post) < 2:
            return messagebox.showerror("Błąd", "Zbyt mało punktów poza obszarem baz, aby znormalizować całkę.")
        I1 = np.polyfit(T[idx_pre], Int_Y[idx_pre], 1)
        Ib1 = np.polyval(I1, T)
        I2 = np.polyfit(T[idx_post], Int_Y[idx_post], 1)
        Ib2 = np.polyval(I2, T)
        alpha = (Int_Y - Ib1) / (Ib2 - Ib1)
        alpha = np.clip(alpha, 0, 1)
        final_baseline = bas1 * (1 - alpha) + bas2 * alpha
        Y_corrected = Y - final_baseline
        tm_index = np.argmax(Y_corrected)
        Tm = T[tm_index]
        delta_cp_at_tm = bas2[tm_index] - bas1[tm_index]

        self.data_state['baseline_analysis'] = pd.DataFrame({'Temp': T, 'MHC': Y, 'Baseline': final_baseline})
        self.data_state['delta_cp_result'] = {'Tm': Tm, 'dCp': delta_cp_at_tm}

        self.update_plot("Wizualizacja Linii Bazowej")
        self.update_button_states()

    def subtract_baseline(self):
        """Odejmuje wcześniej obliczoną linię bazową od danych."""
        df_analysis = self.data_state.get('baseline_analysis')
        if df_analysis is None:
            return messagebox.showerror("Błąd", "Najpierw oblicz i pokaż linię bazową.")

        T = df_analysis['Temp'].values
        Y = df_analysis['MHC'].values
        baseline = df_analysis['Baseline'].values
        Y_corrected = Y - baseline

        self.data_state['peak_only'] = pd.DataFrame({'Temp': T, 'MHC_corr': Y_corrected})

        self.update_plot("Pik po odjęciu linii bazowej")
        self.update_button_states()

    def fit_model(self):
        """Dopasowuje model do piku po odjęciu linii bazowej."""
        df = self.data_state.get('peak_only')
        if df is None:
            return messagebox.showerror("Błąd", "Najpierw odejmij linię bazową.")

        model_name = self.model_var.get()
        if not model_name: return messagebox.showerror("Błąd", "Wybierz model do dopasowania.")

        T = df['Temp'].values
        Cpex = df['MHC_corr'].values * 0.239006  # Konwersja J do cal dla modeli

        fit_function_map = {
            'equilibrium': dsc_models.fit_equilibrium,
            'lumry-eyring': lambda t, y: dsc_models.fit_lumry_eyring(t, y, float(self.param_vars['rate_cpmin'].get())),
            'irreversible': lambda t, y: dsc_models.fit_irreversible(t, y, float(self.param_vars['rate_cpmin'].get()))
        }

        params, metrics, fit_curve_cal = fit_function_map[model_name](T, Cpex)

        if params is None: return  # Błąd dopasowania już został wyświetlony

        fit_curve_J = fit_curve_cal / 0.239006  # Konwersja z powrotem do J
        self.data_state['fit_curve'] = pd.DataFrame({'Temp': T, 'Fit': fit_curve_J})
        self.data_state['final_results'] = {'parameters': params, 'metrics': metrics}

        self.update_plot("Wynik dopasowania modelu")
        self.update_button_states()

    def save_report(self):
        if not any(self.data_state.values()):
            return messagebox.showwarning("Brak danych", "Brak danych do zapisania.")

        filepath = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Plik Excel", "*.xlsx")])
        if not filepath: return

        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                if self.data_state['sample_raw'] is not None:
                    self.data_state['sample_raw'].to_excel(writer, sheet_name='Dane Surowe (Próbka)', index=False)
                if self.data_state['buffer_raw'] is not None:
                    self.data_state['buffer_raw'].to_excel(writer, sheet_name='Dane Surowe (Bufor)', index=False)
                if self.data_state['subtracted'] is not None:
                    self.data_state['subtracted'].to_excel(writer, sheet_name='Po odjęciu bufora', index=False)
                if self.data_state['trimmed'] is not None:
                    self.data_state['trimmed'].to_excel(writer, sheet_name='Po przycięciu', index=False)
                if self.data_state['mhc'] is not None:
                    self.data_state['mhc'].to_excel(writer, sheet_name='Molowa Pojemnosc Cieplna', index=False)
                if self.data_state['baseline_subtracted'] is not None:
                    self.data_state['baseline_subtracted'].to_excel(writer, sheet_name='Po odjęciu bazy', index=False)
                if self.data_state['fit_curve'] is not None:
                    self.data_state['fit_curve'].to_excel(writer, sheet_name='Dopasowany model', index=False)
                if self.data_state['final_results'] is not None:
                    params_df = pd.DataFrame.from_dict(self.data_state['final_results']['parameters'], orient='index',
                                                       columns=['Wartość'])
                    metrics_df = pd.DataFrame.from_dict(self.data_state['final_results']['metrics'], orient='index',
                                                        columns=['Wartość'])
                    pd.concat([params_df, metrics_df]).to_excel(writer, sheet_name='Wyniki Końcowe')
            messagebox.showinfo("Sukces", f"Zapisano raport w pliku:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać raportu: {e}")

    # --- Metody pomocnicze i obsługi zdarzeń ---

    # Wklej w miejsce starej metody on_plot_click
    def on_plot_click(self, event):
        """Obsługuje kliknięcia myszą na wykresie w różnych trybach selekcji."""
        if not self.selection_mode or event.inaxes != self.ax: return

        # --- Logika dla trybu przycinania --- (bez zmian)
        if self.selection_mode == 'trim':
            if event.button == 1:
                self.point_collector.append((event.xdata, event.ydata))
                pt, = self.ax.plot(event.xdata, event.ydata, 'go', markersize=8, markerfacecolor='none')
                self.temp_plot_elements['pre'].append(pt)
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

        # --- Logika dla trybu definicji bazy ---
        if self.selection_mode in ['baseline_pre', 'baseline_post']:
            mode = self.selection_mode.split('_')[1]

            # Lewy przycisk myszy - dodaj punkt
            if event.button == 1 and len(self.point_collector) < 3:
                # ZMIANA: Bardziej jawny sposób rysowania pustego czerwonego kółka
                pt, = self.ax.plot(event.xdata, event.ydata, marker='o',
                                   markerfacecolor='none', markeredgecolor='red',
                                   markersize=8, linestyle='None', markeredgewidth=1.5)
                self.point_collector.append((event.xdata, event.ydata))
                self.temp_plot_elements.setdefault(mode, []).append(pt)
                self._update_temp_baseline_plot(mode)

            # Prawy przycisk myszy - usuń kliknięty punkt
            elif event.button == 3 and self.point_collector:
                points_to_check = self.temp_plot_elements.get(mode, [])
                # Szukamy tylko markerów (obiektów bez stylu linii)
                markers = [el for el in points_to_check if isinstance(el, plt.Line2D) and el.get_linestyle() == 'None']

                if not markers: return

                click_pos_pixels = self.ax.transData.transform((event.xdata, event.ydata))
                point_coords = [marker.get_data() for marker in markers]
                point_pixels = [self.ax.transData.transform(np.squeeze(p)) for p in point_coords]

                distances = [np.linalg.norm(click_pos_pixels - pp) for pp in point_pixels]

                min_dist_idx = np.argmin(distances)
                if distances[min_dist_idx] < 10:  # Tolerancja 10 pikseli
                    # Pobieramy współrzędne usuwanego punktu, aby znaleźć jego indeks w `point_collector`
                    marker_to_remove = markers[min_dist_idx]
                    x_to_remove, y_to_remove = marker_to_remove.get_data()

                    # Znajdź indeks w liście ze współrzędnymi
                    collector_idx = -1
                    for i, (px, py) in enumerate(self.point_collector):
                        if np.isclose(px, x_to_remove) and np.isclose(py, y_to_remove):
                            collector_idx = i
                            break

                    if collector_idx != -1:
                        # Usuń z obu list
                        del self.point_collector[collector_idx]
                        marker_to_remove.remove()
                        self.temp_plot_elements[mode].remove(marker_to_remove)

                        # Przerysuj linię bazową
                        self._update_temp_baseline_plot(mode)

    def update_plot(self, title="", footer_text=""):
        self.clear_temp_plot_elements()
        self.ax.clear()

        # Logika decydująca co narysować, w kolejności od ostatniego etapu
        if self.data_state['fit_curve'] is not None:
            df_peak = self.data_state['peak_only']
            df_fit = self.data_state['fit_curve']
            self.ax.plot(df_peak['Temp'], df_peak['MHC_corr'], 'ro', markersize=3, alpha=0.6, label="Dane")
            self.ax.plot(df_fit['Temp'], df_fit['Fit'], 'b-', linewidth=2, label="Dopasowany model")
            self.ax.legend()
        elif self.data_state.get('peak_only') is not None:
            df = self.data_state['peak_only']
            self.ax.plot(df['Temp'], df['MHC_corr'], 'b-', label="Sygnał po odjęciu bazy")
            self.ax.legend()
        elif self.data_state.get('baseline_analysis') is not None:
            df = self.data_state['baseline_analysis']
            self.ax.plot(df['Temp'], df['MHC'], label="Molowa Poj. Cieplna")
            self.ax.plot(df['Temp'], df['Baseline'], 'r--', label="Obliczona linia bazowa")
            # Dodanie pola tekstowego z DeltaCp
            dcp_res = self.data_state['delta_cp_result']
            text_str = f"Tm = {dcp_res['Tm']:.2f} °C\nΔCp = {dcp_res['dCp']:.2f} J/mol·K"
            self.ax.text(0.05, 0.95, text_str, transform=self.ax.transAxes, fontsize=10,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            self.ax.legend()
        elif self.data_state['mhc'] is not None:
            df = self.data_state['mhc']
            self.ax.plot(df['Temp'], df['MHC'], label="Molowa Poj. Cieplna")
            self.ax.legend()
        elif self.data_state['trimmed'] is not None:
            df = self.data_state['trimmed']
            self.ax.plot(df['Temp'], df['Signal'], label="Sygnał po przycięciu")
            self.ax.legend()
        elif self.data_state['subtracted'] is not None:
            df = self.data_state['subtracted']
            self.ax.plot(df['Temp'], df['Signal'], label="Sygnał po odjęciu bufora")
            self.ax.legend()
        elif self.data_state['sample_raw'] is not None:
            df = self.data_state['sample_raw']
            self.ax.plot(df['Temp'], df['Signal'], label="Surowy sygnał (próbka)")
            if self.data_state['buffer_raw'] is not None:
                df_buf = self.data_state['buffer_raw']
                self.ax.plot(df_buf['Temp'], df_buf['Signal'], label="Surowy sygnał (bufor)", alpha=0.7)
            self.ax.legend()

        self.ax.set_title(title)
        self.ax.set_xlabel("Temperatura [°C]")
        self.ax.set_ylabel("Sygnał")
        self.ax.grid(True)
        self.fig.text(0.5, 0.01, footer_text, ha='center', color='blue', fontsize=9)
        self.fig.tight_layout(rect=[0, 0.03, 1, 1])
        self.canvas.draw()

    def clear_temp_plot_elements(self, key=None):
        """Czyści tymczasowe elementy z wykresu."""
        if key:
            # Czyści elementy tylko dla danego klucza (np. 'pre' lub 'post')
            for element in self.temp_plot_elements.get(key, []):
                element.remove()
            self.temp_plot_elements[key] = []
        else:
            # Czyści wszystkie tymczasowe elementy
            for key_ in self.temp_plot_elements:
                for element in self.temp_plot_elements[key_]:
                    element.remove()
                self.temp_plot_elements[key_] = []

        self.canvas.draw() if hasattr(self, 'canvas') else None

    # Wklej w miejsce starej metody _update_temp_baseline_plot
    # Wklej w miejsce starej metody _update_temp_baseline_plot
    def _update_temp_baseline_plot(self, mode):
        """Aktualizuje tymczasowy wykres linii bazowej (liniowej dla 2 pkt, kwadratowej dla 3)."""
        # KROK 1: Usuń starą linię przerywaną, ale zostaw punkty (markery).
        # ZMIANA: Wyszukujemy obiekty, które są liniami (mają styl '--'), a nie markerami (których styl linii to 'None').
        # To jest kluczowa poprawka.
        elements_in_mode = self.temp_plot_elements.get(mode, [])
        old_lines = [el for el in elements_in_mode if isinstance(el, plt.Line2D) and el.get_linestyle() != 'None']
        for line in old_lines:
            line.remove()
            self.temp_plot_elements[mode].remove(line)

        points = np.array(self.point_collector)
        if len(points) < 2:
            self.canvas.draw()
            return

        # KROK 2: Dopasuj model (bez zmian)
        points = points[points[:, 0].argsort()]
        poly_deg = 1 if len(points) == 2 else 2
        p = np.polyfit(points[:, 0], points[:, 1], poly_deg)

        # KROK 3: Narysuj nową linię przerywaną (bez zmian)
        x_fit = np.linspace(points[0, 0], points[-1, 0], 100)
        y_fit = np.polyval(p, x_fit)
        line, = self.ax.plot(x_fit, y_fit, 'r--')
        self.temp_plot_elements[mode].append(line)

        # KROK 4: Zapisz parametry (bez zmian)
        if len(points) == 3:
            self.data_state['baseline_params'][mode] = {'points': points, 'poly': p}
            self.update_button_states()

        self.canvas.draw()

    def update_button_states(self):
        """Włącza/wyłącza przyciski w zależności od stanu analizy."""
        s = self.data_state

        # Warunki włączania/wyłączania przycisków
        can_subtract_buffer = s['sample_raw'] is not None and s['buffer_raw'] is not None
        can_trim = s['subtracted'] is not None
        can_convert = s['trimmed'] is not None
        can_define_base = s['mhc'] is not None
        can_show_base = 'pre' in s['baseline_params'] and 'post' in s['baseline_params']
        # Nowy stan "baseline_analysis" będzie tworzony przez funkcję 'show_baseline'
        can_subtract_base = s.get('baseline_analysis') is not None
        # Nowy stan "peak_only" będzie tworzony przez funkcję 'subtract_baseline'
        can_fit = s.get('peak_only') is not None
        can_save = any(value is not None for value in s.values())

        # Ustawianie stanu przycisków
        self.btn_subtract['state'] = 'normal' if can_subtract_buffer else 'disabled'
        self.btn_trim['state'] = 'normal' if can_trim else 'disabled'
        self.btn_convert['state'] = 'normal' if can_convert else 'disabled'
        self.btn_base_pre['state'] = 'normal' if can_define_base else 'disabled'
        self.btn_base_post['state'] = 'normal' if can_define_base else 'disabled'
        self.btn_show_base['state'] = 'normal' if can_show_base else 'disabled'
        self.btn_subtract_base['state'] = 'normal' if can_subtract_base else 'disabled'
        self.btn_fit['state'] = 'normal' if can_fit else 'disabled'
        self.btn_save['state'] = 'normal' if can_save else 'disabled'

if __name__ == "__main__":
    app = DSCAnalyzerApp()
    app.mainloop()