# data_generator_gui.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Importujemy model z naszego wcześniej stworzonego pliku
try:
    from dsc_models import model_equilibrium
except ImportError:
    messagebox.showerror(
        "Błąd importu",
        "Nie można znaleźć pliku 'dsc_models.py'.\n"
        "Upewnij się, że ten skrypt znajduje się w tym samym folderze co 'dsc_models.py'."
    )
    exit()


class DataGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Generator Danych DSC")
        self.geometry("1200x700")

        self.generated_data = None

        # --- Główny kontener ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Panel sterowania (lewa strona) ---
        control_frame = ttk.Frame(main_frame, width=350)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)  # Zapobiega kurczeniu się ramki

        # --- Panel wykresu (prawa strona) ---
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._create_controls(control_frame)
        self._create_plot()

    def _create_controls(self, parent):
        """Tworzy widżety do wprowadzania parametrów."""
        # Używamy słownika do przechowywania zmiennych
        self.param_vars = {}

        # --- Parametry modelu (pik) ---
        # Zmieniono jednostki na J/mol i J/mol·K, aby były zgodne z dsc_models.py
        model_frame = ttk.LabelFrame(parent, text="Parametry Piku (Model Równowagowy)", padding="10")
        model_frame.pack(fill=tk.X, pady=5)

        params_model = {
            "dH (J/mol)": "418400.0",  # Odpowiednik 100000 cal/mol
            "Tm (°C)": "70.0",
            "dCp (J/mol·K)": "6276.0"  # Odpowiednik 1500 cal/mol·K
        }
        for i, (text, default) in enumerate(params_model.items()):
            label = ttk.Label(model_frame, text=text)
            label.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(self, value=default)
            entry = ttk.Entry(model_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
            # Klucz to np. 'dH', ale bez jednostek, aby pasował do argumentów funkcji
            self.param_vars[text.split(" ")[0]] = var

        # --- Parametry Linii Bazowej i Szumu ---
        base_frame = ttk.LabelFrame(parent, text="Linia Bazowa i Szum", padding="10")
        base_frame.pack(fill=tk.X, pady=5)

        params_base = {
            "Offset (J/mol·K)": "5.0", # Zmieniono etykietę na J/mol.K dla spójności
            "Nachylenie liniowe (J/mol·K²/°C)": "0.1", # Doprecyzowano jednostki
            "Szum (std dev) (J/mol·K)": "0.5" # Doprecyzowano jednostki
        }
        for i, (text, default) in enumerate(params_base.items()):
            label = ttk.Label(base_frame, text=text)
            label.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(self, value=default)
            entry = ttk.Entry(base_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
            # Klucz to np. 'Offset' - używamy pierwszej części nazwy
            self.param_vars[text.split(" ")[0]] = var

        # --- Zakres danych ---
        range_frame = ttk.LabelFrame(parent, text="Zakres Danych", padding="10")
        range_frame.pack(fill=tk.X, pady=5)

        params_range = {
            "Temp. min (°C)": "20.0",
            "Temp. max (°C)": "120.0"
        }
        for i, (text, default) in enumerate(params_range.items()):
            label = ttk.Label(range_frame, text=text)
            label.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(self, value=default)
            entry = ttk.Entry(range_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
            # Zmieniono klucze, aby usunąć spacje i znaki specjalne dla łatwiejszego dostępu
            self.param_vars[text.replace('.', '').replace(' (°C)', '').replace(' ', '_').lower()] = var

        # --- Przyciski ---
        button_frame = ttk.Frame(parent, padding="10")
        button_frame.pack(fill=tk.X, pady=20)

        gen_button = ttk.Button(button_frame, text="Generuj", command=self.generate_and_plot_data)
        gen_button.pack(fill=tk.X, pady=5)

        save_button = ttk.Button(button_frame, text="Zapisz CSV", command=self.save_csv)
        save_button.pack(fill=tk.X, pady=5)

    def _create_plot(self):
        """Tworzy pusty wykres Matplotlib w ramce."""
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.ax.set_title("Wygenerowany Termogram DSC")
        self.ax.set_xlabel("Temperatura [°C]")
        # Zmieniono etykietę Y na bardziej specyficzną
        self.ax.set_ylabel("Pojemność cieplna [J/(mol·K)] (arbitralne jednostki)")
        self.ax.grid(True)
        self.fig.tight_layout()

    def generate_and_plot_data(self):
        """Pobiera parametry, generuje dane i aktualizuje wykres."""
        try:
            # Pobierz wartości z pól tekstowych
            p = {key: float(var.get()) for key, var in self.param_vars.items()}

            # Stwórz wektor temperatury
            T = np.linspace(p['temp_min'], p['temp_max'], 1000)

            # Generuj sygnał z modelu równowagowego (bez dodatkowej linii bazowej)
            # Parametr 'baseline' w modelu to nasz 'Offset'
            ideal_signal = model_equilibrium(
                T_C=T,
                dH=p['dH'],
                Tm=p['Tm'],
                dCp=p['dCp'],
                baseline=p['Offset']
            )

            # Dodaj dodatkowe nachylenie liniowe
            # Używamy klucza 'Nachylenie' z pola wejściowego
            linear_baseline_component = p['Nachylenie'] * (T - p['temp_min'])
            signal_with_slope = ideal_signal + linear_baseline_component

            # Dodaj szum
            noise = np.random.normal(0, p['Szum'], len(T))
            final_signal = signal_with_slope + noise

            # Zapisz dane do atrybutu klasy
            self.generated_data = np.vstack((T, final_signal)).T

            # Aktualizuj wykres
            self.ax.clear()
            self.ax.plot(T, final_signal, 'b-', label='Sygnał z szumem')
            self.ax.plot(T, signal_with_slope, 'r--', label='Idealny sygnał (z linią bazową)', alpha=0.7)
            self.ax.set_title("Wygenerowany Termogram DSC")
            self.ax.set_xlabel("Temperatura [°C]")
            # Zmieniono etykietę Y na bardziej specyficzną
            self.ax.set_ylabel("Pojemność cieplna [J/(mol·K)] (arbitralne jednostki)")
            self.ax.grid(True)
            self.ax.legend()
            self.fig.tight_layout()
            self.canvas.draw()

        except ValueError:
            messagebox.showerror("Błąd wartości",
                                 "Proszę wprowadzić prawidłowe wartości liczbowe we wszystkich polach.")
        except Exception as e:
            messagebox.showerror("Wystąpił błąd", str(e))

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

        if not filepath:
            return  # Użytkownik anulował

        try:
            np.savetxt(
                filepath,
                self.generated_data,
                delimiter=",",
                header="Temperatura,Sygnal",
                comments=""
            )
            messagebox.showinfo("Sukces", f"Pomyślnie zapisano dane w pliku:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać pliku.\nBłąd: {str(e)}")


if __name__ == "__main__":
    app = DataGeneratorApp()
    app.mainloop()