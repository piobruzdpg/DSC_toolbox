import tkinter as tk
from tkinter import filedialog, messagebox, Listbox
import pandas as pd
import pytainanodsc
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os


class DscApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Przeglądarka plików NanoDSC (.dsc)")
        self.root.geometry("1000x600")

        # Zmienne przechowujące stan
        self.dsc_data = None
        self.current_filepath = None

        # --- Struktura GUI ---

        # Ramka główna
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel lewy (lista skanów)
        left_frame = tk.Frame(main_frame, width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)  # Zapobiega kurczeniu się ramki

        tk.Label(left_frame, text="Dostępne skany:", font=("Helvetica", 12)).pack(anchor="w")
        self.scan_listbox = Listbox(left_frame, selectmode=tk.EXTENDED)  # EXTENDED pozwala na wielokrotne zaznaczenie
        self.scan_listbox.pack(fill=tk.BOTH, expand=True)
        self.scan_listbox.bind('<<ListboxSelect>>', self.on_scan_select)

        # Panel prawy (wykres)
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(right_frame, text="Podgląd skanu:", font=("Helvetica", 12)).pack(anchor="w")

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Panel dolny (przyciski)
        bottom_frame = tk.Frame(root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.open_button = tk.Button(bottom_frame, text="Otwórz plik .dsc", command=self.open_file)
        self.open_button.pack(side=tk.LEFT, padx=(0, 5))

        self.export_button = tk.Button(bottom_frame, text="Eksportuj zaznaczone do CSV",
                                       command=self.export_selected_scans_to_csv)
        self.export_button.pack(side=tk.LEFT)

    def open_file(self):
        """Otwiera okno dialogowe do wyboru pliku .dsc i wczytuje dane."""
        filepath = filedialog.askopenfilename(
            title="Wybierz plik .dsc",
            filetypes=(("Pliki DSC", "*.dsc"), ("Wszystkie pliki", "*.*"))
        )
        if not filepath:
            return

        try:
            self.current_filepath = filepath
            self.dsc_data = pytainanodsc.load(filepath)
            self.update_scan_list()
            self.ax.clear()
            self.ax.set_title("Wybierz skan z listy")
            self.ax.set_xlabel("Temperatura (°C)")
            self.ax.set_ylabel("Moc (µJ/s)")
            self.canvas.draw()
            self.root.title(f"Przeglądarka plików NanoDSC - {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Błąd odczytu pliku", f"Nie udało się wczytać pliku: {e}")

    def update_scan_list(self):
        """Aktualizuje listę skanów w GUI."""
        self.scan_listbox.delete(0, tk.END)
        if self.dsc_data:
            for i, scan in enumerate(self.dsc_data.scans):
                scan_label = f"Skan {scan.scan_number} ({scan.scan_type})"
                self.scan_listbox.insert(tk.END, scan_label)

    def on_scan_select(self, event):
        """Wywoływana po wybraniu skanu z listy, aktualizuje wykres."""
        selected_indices = self.scan_listbox.curselection()
        if not selected_indices:
            return

        # Bierzemy pierwszy zaznaczony element do podglądu
        selected_index = selected_indices[0]
        scan = self.dsc_data.scans[selected_index]

        # Konwersja danych
        df = self.convert_scan_to_dataframe(scan)

        # Rysowanie wykresu
        self.ax.clear()
        self.ax.plot(df['Temperatura (°C)'], df['Moc (µJ/s)'])
        self.ax.set_title(f"Podgląd: Skan {scan.scan_number} ({scan.scan_type})")
        self.ax.set_xlabel("Temperatura (°C)")
        self.ax.set_ylabel("Moc (µJ/s)")
        self.ax.grid(True)
        self.fig.tight_layout()
        self.canvas.draw()

    def export_selected_scans_to_csv(self):
        """Eksportuje wszystkie zaznaczone skany do plików CSV."""
        selected_indices = self.scan_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Brak zaznaczenia", "Proszę zaznaczyć przynajmniej jeden skan do eksportu.")
            return

        # Pytamy użytkownika o folder docelowy
        output_dir = filedialog.askdirectory(title="Wybierz folder do zapisu plików CSV")
        if not output_dir:
            return

        count = 0
        for index in selected_indices:
            scan = self.dsc_data.scans[index]
            df_to_export = self.convert_scan_to_dataframe(scan)

            # Tworzenie nazwy pliku
            base_filename = os.path.splitext(os.path.basename(self.current_filepath))[0]
            output_filename = f"{base_filename}_skan_{scan.scan_number}_{scan.scan_type}.csv"
            output_path = os.path.join(output_dir, output_filename)

            # Zapis do CSV
            df_to_export.to_csv(output_path, index=False, sep=';', decimal='.')
            count += 1

        messagebox.showinfo("Eksport zakończony", f"Pomyślnie wyeksportowano {count} plików do folderu:\n{output_dir}")

    def convert_scan_to_dataframe(self, scan):
        """Konwertuje obiekt skanu na DataFrame w docelowych jednostkach."""
        df_kelvin = scan.to_dataframe()
        final_df = pd.DataFrame({
            'Temperatura (°C)': df_kelvin['temperature'] - 273.15,
            'Moc (µJ/s)': df_kelvin['power'] * 1_000_000
        })
        return final_df


if __name__ == "__main__":
    root = tk.Tk()
    app = DscApp(root)
    root.mainloop()