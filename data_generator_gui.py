# data_generator_gui.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.integrate import cumulative_trapezoid  # <-- NOWY IMPORT

# Importujemy modele z naszego wcześniej stworzonego pliku
try:
    from dsc_models import model_equilibrium, model_lumry_eyring, model_irreversible
except ImportError:
    messagebox.showerror(
        "Błąd importu",
        "Nie można znaleźć pliku 'dsc_models.py'.\n"
        "Upewnij się, że ten skrypt znajduje się w tym samym folderze co 'dsc_models.py'."
    )
    exit()


class DataGeneratorApp(tk.Tk):
    """
    Aplikacja GUI do generowania syntetycznych danych DSC na podstawie
    różnych modeli termodynamicznych, z uwzględnieniem zmiany linii bazowej (ΔCp).
    """

    def __init__(self):
        super().__init__()
        self.title("Generator Danych DSC")
        self.geometry("1200x750")

        self.generated_data = None
        self.param_vars = {}

        # Słownik definiujący parametry dla każdego modelu
        # Format: { 'Nazwa wyświetlana': ('klucz_wewnętrzny', 'wartość_domyślna') }
        self.MODEL_DEFINITIONS = {
            'equilibrium': {
                'dH (J/mol)': ('dH', '450000.0'),
                'Tm (°C)': ('Tm', '75.0'),
            },
            'lumry-eyring': {
                'dH (J/mol)': ('dH', '450000.0'),
                'Tm (°C)': ('Tm', '75.0'),
                'Ea (J/mol)': ('Ea', '600000.0'),
                'T* (°C)': ('Ta', '85.0'),
            },
            'irreversible': {
                'dH (J/mol)': ('dH', '450000.0'),
                'Ea (J/mol)': ('Ea', '600000.0'),
                'T* (°C)': ('Ta', '80.0'),
            }
        }

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
        self._create_plot()
        self._update_parameter_widgets()  # Inicjalizacja pól dla domyślnego modelu

    def _create_controls(self, parent):
        """Tworzy wszystkie widżety w panelu sterowania."""
        # --- Wybór modelu ---
        model_selection_frame = ttk.LabelFrame(parent, text="1. Wybór Modelu", padding="10")
        model_selection_frame.pack(fill=tk.X, pady=5)

        self.model_var = tk.StringVar(value='equilibrium')
        model_combo = ttk.Combobox(model_selection_frame, textvariable=self.model_var, state="readonly",
                                   values=list(self.MODEL_DEFINITIONS.keys()))
        model_combo.pack(fill=tk.X)
        model_combo.bind("<<ComboboxSelected>>", self._update_parameter_widgets)

        # --- Parametry modelu (dynamiczne) ---
        self.model_params_frame = ttk.LabelFrame(parent, text="2. Parametry Piku", padding="10")
        self.model_params_frame.pack(fill=tk.X, pady=5)

        # --- Parametry skanowania (dla modeli kinetycznych) ---
        self.scan_params_frame = ttk.LabelFrame(parent, text="3. Parametry Skanowania", padding="10")
        self.scan_params_frame.pack(fill=tk.X, pady=5)
        label = ttk.Label(self.scan_params_frame, text="Szybkość grzania (°C/min)")
        label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.param_vars['heating_rate'] = tk.StringVar(self, value="1.0")
        entry = ttk.Entry(self.scan_params_frame, textvariable=self.param_vars['heating_rate'], width=15)
        entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)

        # --- ZMODYFIKOWANA SEKCJA: Parametry Linii Bazowej i Szumu ---
        base_frame = ttk.LabelFrame(parent, text="4. Linia Bazowa i Szum", padding="10")
        base_frame.pack(fill=tk.X, pady=5)
        base_params = {
            "Offset (jedn. sygnału)": ('offset', '2000.0'),
            "Nachylenie (jedn./°C)": ('slope', '50.0'),
            "ΔCp (J/mol·K)": ('delta_cp', '5000.0'),  # <-- NOWA LINIA
            "Szum (std dev)": ('noise_std', '100.0')
        }
        for i, (text, (key, default)) in enumerate(base_params.items()):
            label = ttk.Label(base_frame, text=text)
            label.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(self, value=default)
            entry = ttk.Entry(base_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
            self.param_vars[key] = var

        # --- Zakres danych ---
        range_frame = ttk.LabelFrame(parent, text="5. Zakres Danych", padding="10")
        range_frame.pack(fill=tk.X, pady=5)
        range_params = {"Temp. min (°C)": ('temp_min', '20.0'), "Temp. max (°C)": ('temp_max', '120.0')}
        for i, (text, (key, default)) in enumerate(range_params.items()):
            label = ttk.Label(range_frame, text=text)
            label.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(self, value=default)
            entry = ttk.Entry(range_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
            self.param_vars[key] = var

        # --- Przyciski ---
        button_frame = ttk.Frame(parent, padding="10")
        button_frame.pack(fill=tk.X, pady=20)
        gen_button = ttk.Button(button_frame, text="Generuj i Rysuj", command=self.generate_and_plot_data)
        gen_button.pack(fill=tk.X, pady=5)
        save_button = ttk.Button(button_frame, text="Zapisz do CSV", command=self.save_csv)
        save_button.pack(fill=tk.X, pady=5)

    def _update_parameter_widgets(self, event=None):
        """Czyści i ponownie tworzy widżety parametrów dla wybranego modelu."""
        for widget in self.model_params_frame.winfo_children():
            widget.destroy()

        model_name = self.model_var.get()
        params_def = self.MODEL_DEFINITIONS.get(model_name, {})

        for i, (text, (key, default)) in enumerate(params_def.items()):
            label = ttk.Label(self.model_params_frame, text=text)
            label.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(self, value=default)
            entry = ttk.Entry(self.model_params_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
            self.param_vars[key] = var

        if model_name in ['lumry-eyring', 'irreversible']:
            self.scan_params_frame.pack(fill=tk.X, pady=5, after=self.model_params_frame)
        else:
            self.scan_params_frame.pack_forget()

    def _create_plot(self):
        """Tworzy pusty wykres Matplotlib."""
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.ax.set_title("Wygenerowany Termogram DSC")
        self.ax.set_xlabel("Temperatura [°C]")
        self.ax.set_ylabel("Pojemność cieplna [J/(mol·K)]")
        self.ax.grid(True)
        self.fig.tight_layout()

    def generate_and_plot_data(self):
        """Pobiera parametry, generuje dane z sigmoidalną bazą i aktualizuje wykres."""
        try:
            p = {key: float(var.get()) for key, var in self.param_vars.items() if var.get()}
            model_name = self.model_var.get()

            T = np.linspace(p['temp_min'], p['temp_max'], 1500)

            # --- KROK 1: Generowanie "idealnego" piku na zerowej linii bazowej ---
            ideal_peak = None
            if model_name == 'equilibrium':
                ideal_peak = model_equilibrium(T_C=T, dH=p['dH'], Tm=p['Tm'])
            elif model_name == 'lumry-eyring':
                ideal_peak = model_lumry_eyring(T_C=T, dH=p['dH'], Tm=p['Tm'], Ea=p['Ea'], Ta=p['Ta'],
                                                heating_rate_C_min=p['heating_rate'])
            elif model_name == 'irreversible':
                ideal_peak = model_irreversible(T_C=T, dH=p['dH'], Ea=p['Ea'], Ta=p['Ta'],
                                                heating_rate_C_min=p['heating_rate'])

            if ideal_peak is None:
                raise ValueError("Nieznany model lub błąd generowania piku.")

            # --- KROK 2: Tworzenie sigmoidalnej linii bazowej na podstawie kształtu piku ---
            # Obliczamy całkę skumulowaną z piku, aby uzyskać kształt sigmoidy
            integral = cumulative_trapezoid(ideal_peak, T, initial=0)
            # Normalizujemy ją do zakresu [0, 1], aby użyć jako czynnika wagowego
            min_integral, max_integral = np.min(integral), np.max(integral)
            if np.isclose(max_integral, min_integral):
                transition_factor = np.zeros_like(T)
            else:
                transition_factor = (integral - min_integral) / (max_integral - min_integral)

            # Definiujemy linię bazową przed i po piku
            base_pre_transition = p['offset'] + p['slope'] * (T - p['temp_min'])
            # Budujemy finalną sigmoidalną linię bazową, która zmienia się o ΔCp
            sigmoidal_baseline = base_pre_transition + transition_factor * p['delta_cp']

            # --- KROK 3: Łączenie piku, linii bazowej i szumu ---
            signal_without_noise = ideal_peak + sigmoidal_baseline
            noise = np.random.normal(0, p['noise_std'], len(T))
            final_signal = signal_without_noise + noise

            # Zapisz dane do atrybutu klasy
            self.generated_data = np.vstack((T, final_signal)).T

            # Aktualizuj wykres
            self.ax.clear()
            self.ax.plot(T, final_signal, 'b-', label='Sygnał (pik + baza + szum)')
            self.ax.plot(T, signal_without_noise, 'r--', label='Idealny sygnał (pik + baza)', alpha=0.8)
            self.ax.set_title(f"Wygenerowany Termogram DSC (Model: {model_name})")
            self.ax.set_xlabel("Temperatura [°C]")
            self.ax.set_ylabel("Pojemność cieplna [J/(mol·K)]")
            self.ax.grid(True)
            self.ax.legend()
            self.fig.tight_layout()
            self.canvas.draw()

        except ValueError:
            messagebox.showerror("Błąd wartości",
                                 "Proszę wprowadzić prawidłowe wartości liczbowe we wszystkich polach.")
        except Exception as e:
            messagebox.showerror("Wystąpił błąd", f"Nie udało się wygenerować danych:\n{str(e)}")

    def save_csv(self):
        """Zapisuje wygenerowane dane do pliku CSV."""
        if self.generated_data is None:
            messagebox.showwarning("Brak danych", "Najpierw wygeneruj dane za pomocą przycisku 'Generuj'.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("Pliki CSV", "*.csv"), ("Wszystkie pliki", "*.*")],
            title="Zapisz dane jako..."
        )
        if not filepath: return

        try:
            np.savetxt(filepath, self.generated_data, delimiter=",", header="Temp,Signal", comments="")
            messagebox.showinfo("Sukces", f"Pomyślnie zapisano dane w pliku:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać pliku.\nBłąd: {str(e)}")


if __name__ == "__main__":
    app = DataGeneratorApp()
    app.mainloop()