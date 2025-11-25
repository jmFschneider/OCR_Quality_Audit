import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class InteractionGUI:
    """
    Fenêtre pour l'analyse des interactions de 2ème ordre via un graphique de contour.
    """
    def __init__(self, master, df):
        self.top = tk.Toplevel(master)
        self.top.title("Analyse d'Interaction des Paramètres (2D)")
        self.top.geometry("800x750")

        self.df = df
        self.numeric_cols = list(df.select_dtypes(include=['float64', 'int64']).columns)
        
        params_to_exclude = ['score_tesseract', 'score_nettete', 'trial_id']
        self.param_options = [col for col in self.numeric_cols if col not in params_to_exclude]

        if len(self.param_options) < 2:
            messagebox.showerror("Erreur", "Pas assez de paramètres numériques (< 2) pour une analyse d'interaction.", parent=self.top)
            self.top.destroy()
            return
            
        self.param1_var = tk.StringVar(value=self.param_options[0])
        self.param2_var = tk.StringVar(value=self.param_options[1])

        self.create_widgets()

    def create_widgets(self):
        selector_frame = ttk.LabelFrame(self.top, text="Sélection des Paramètres")
        selector_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(selector_frame, text="Paramètre 1 (Axe X):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.p1_combo = ttk.Combobox(selector_frame, textvariable=self.param1_var, state="readonly", values=self.param_options)
        self.p1_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(selector_frame, text="Paramètre 2 (Axe Y):").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.p2_combo = ttk.Combobox(selector_frame, textvariable=self.param2_var, state="readonly", values=self.param_options)
        self.p2_combo.grid(row=0, column=3, padx=5, pady=5)
        
        plot_btn = ttk.Button(selector_frame, text="🎨 Générer le graphique lissé", command=self.plot_interaction)
        plot_btn.grid(row=1, column=0, columnspan=4, pady=10)

        self.plot_frame = ttk.Frame(self.top)
        self.plot_frame.pack(padx=10, pady=10, fill="both", expand=True)

    def plot_interaction(self):
        param1 = self.param1_var.get()
        param2 = self.param2_var.get()

        if param1 == param2:
            messagebox.showwarning("Sélection Invalide", "Veuillez choisir deux paramètres différents.", parent=self.top)
            return

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(8, 7))

        # --- Lissage par interpolation cubique sur une grille ---
        # 1. Définir une grille régulière
        grid_x, grid_y = np.mgrid[
            self.df[param1].min():self.df[param1].max():100j, 
            self.df[param2].min():self.df[param2].max():100j
        ]

        # 2. Préparer les points et valeurs pour l'interpolation
        points = self.df[[param1, param2]].values
        values = self.df['score_tesseract'].values

        # 3. Interpoler avec la méthode cubique pour un résultat lisse
        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

        # 4. Dessiner le contour à partir de la grille lissée
        contour = ax.contourf(grid_x, grid_y, grid_z, levels=15, cmap='viridis')
        
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Score Tesseract (interpolé)')

        # Superposer les points de données réels pour comparaison
        ax.scatter(self.df[param1], self.df[param2], c='red', s=10, label='Essais effectués')

        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_title(f"Interaction lissée entre {param1} et {param2} sur le Score Tesseract")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()


class CorrelationGUI:
    """
    Cette classe gère la fenêtre d'analyse de corrélation.
    """
    def __init__(self, master, df):
        self.top = tk.Toplevel(master)
        self.top.title("Analyse de Corrélation avec le Score Tesseract")
        self.top.geometry("800x600")

        if 'score_tesseract' not in df.columns:
            messagebox.showerror("Erreur", "La colonne 'score_tesseract' est introuvable.", parent=self.top)
            self.top.destroy()
            return

        df_numeric = df.select_dtypes(include=['float64', 'int64'])

        if 'score_tesseract' not in df_numeric.columns:
            messagebox.showerror("Erreur", "La colonne 'score_tesseract' n'est pas de type numérique.", parent=self.top)
            self.top.destroy()
            return

        corr_series = df_numeric.corr()['score_tesseract'].sort_values(ascending=True)
        corr_series = corr_series.drop('score_tesseract', errors='ignore')

        if corr_series.empty:
            messagebox.showinfo("Information", "Aucune autre colonne numérique à corréler avec 'score_tesseract'.", parent=self.top)
            self.top.destroy()
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#d62728' if c < 0 else '#2ca02c' for c in corr_series.values]
        corr_series.plot(kind='barh', ax=ax, color=colors)
        ax.set_title("Corrélation des Paramètres avec le Score Tesseract")
        ax.set_xlabel("Coefficient de Corrélation de Pearson")
        ax.set_ylabel("Paramètres")
        ax.axvline(0, color='grey', linewidth=0.8)
        ax.grid(True, linestyle='--', alpha=0.6, axis='x')

        for index, value in enumerate(corr_series):
            ax.text(value, index, f' {value:.3f}', va='center', ha='left' if value >= 0 else 'right')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.top)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self.top)
        toolbar.update()


class AnalyticsGUI:
    def __init__(self, master):
        self.master = master
        master.title("📊 Analyse Statistique des Optimisations")
        master.geometry("900x700")
        self.df = None
        self.column_options = ["N/A - Charger fichier..."]
        self.x_axis_var = tk.StringVar(value=self.column_options[0])
        self.y1_axis_var = tk.StringVar(value=self.column_options[0])
        self.y2_axis_var = tk.StringVar(value=self.column_options[0])
        self.create_widgets()

    def create_widgets(self):
        control_frame = ttk.LabelFrame(self.master, text="Contrôles d'Analyse")
        control_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(control_frame, text="Fichier de Résultats :").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.file_label = ttk.Label(control_frame, text="Aucun fichier chargé.", width=40, anchor="w")
        self.file_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.load_button = ttk.Button(control_frame, text="💾 Charger CSV", command=self.load_file)
        self.load_button.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(control_frame, text="Axe X (Paramètre) :").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.x_combo = ttk.Combobox(control_frame, textvariable=self.x_axis_var, state="readonly", values=self.column_options, width=35)
        self.x_combo.grid(row=1, column=1, padx=5, pady=5, columnspan=2, sticky="ew")

        ttk.Label(control_frame, text="Axe Y1 (Score 1) :").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.y1_combo = ttk.Combobox(control_frame, textvariable=self.y1_axis_var, state="readonly", values=self.column_options, width=35)
        self.y1_combo.grid(row=2, column=1, padx=5, pady=5, columnspan=2, sticky="ew")

        ttk.Label(control_frame, text="Axe Y2 (Score 2) :").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.y2_combo = ttk.Combobox(control_frame, textvariable=self.y2_axis_var, state="readonly", values=self.column_options, width=35)
        self.y2_combo.grid(row=3, column=1, padx=5, pady=5, columnspan=2, sticky="ew")

        self.plot_button = ttk.Button(control_frame, text="📈 Mettre à jour le Graphique", command=self.update_plot)
        self.plot_button.grid(row=4, column=0, columnspan=3, pady=10, padx=5, sticky="ew")
        
        self.corr_button = ttk.Button(control_frame, text="🔬 Analyser les Corrélations", command=self.open_correlation_window)
        self.corr_button.grid(row=5, column=0, columnspan=3, pady=5, padx=5, sticky="ew")
        
        self.interaction_button = ttk.Button(control_frame, text="🎨 Analyser les Interactions (2D)", command=self.open_interaction_window)
        self.interaction_button.grid(row=6, column=0, columnspan=3, pady=5, padx=5, sticky="ew")

        plot_frame = ttk.Frame(self.master)
        plot_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.fig, self.ax1 = plt.subplots(figsize=(8, 6))
        self.ax2 = self.ax1.twinx()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

        self.x_axis_var.trace_add("write", lambda *args: self.update_plot())
        self.y1_axis_var.trace_add("write", lambda *args: self.update_plot())
        self.y2_axis_var.trace_add("write", lambda *args: self.update_plot())

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not file_path: return

        try:
            self.df = pd.read_csv(file_path, sep=';')
            self.df = self.df.apply(pd.to_numeric, errors='ignore')
            self.file_label.config(text=file_path.split('/')[-1])
            self.column_options = ["(aucun)"] + list(self.df.columns)
            
            for combo in [self.x_combo, self.y1_combo, self.y2_combo]:
                combo['values'] = self.column_options

            if 'trial_id' in self.df.columns: self.x_axis_var.set('trial_id')
            if 'score_tesseract' in self.df.columns: self.y1_axis_var.set('score_tesseract')
            self.y2_axis_var.set("(aucun)")
            if 'score_nettete' in self.df.columns: self.y2_axis_var.set('score_nettete')

            self.update_plot()
        except Exception as e:
            messagebox.showerror("Erreur de Chargement", f"Impossible de lire le fichier :\n{e}")
            self.df = None

    def open_correlation_window(self):
        if self.df is None:
            messagebox.showwarning("Aucune Donnée", "Veuillez d'abord charger un fichier CSV.")
            return
        CorrelationGUI(self.master, self.df)
        
    def open_interaction_window(self):
        if self.df is None:
            messagebox.showwarning("Aucune Donnée", "Veuillez d'abord charger un fichier CSV.")
            return
        InteractionGUI(self.master, self.df)

    def update_plot(self):
        if self.df is None: return
        x_col, y1_col, y2_col = self.x_axis_var.get(), self.y1_axis_var.get(), self.y2_axis_var.get()

        self.ax1.cla()
        self.ax2.cla()
        self.ax2.set_visible(False)

        if x_col == '(aucun)': return

        # Tracé Y1
        if y1_col != "(aucun)" and y1_col in self.df.columns:
            self.ax1.plot(self.df[x_col], self.df[y1_col], label=y1_col, color='#1f77b4', marker='*', markersize=5, linestyle='') # Correction: linestyle=''
            self.ax1.set_ylabel(y1_col.replace('_', ' ').title(), color='#1f77b4')
            self.ax1.tick_params(axis='y', labelcolor='#1f77b4')
            self.ax1.spines['left'].set_color('#1f77b4')

        # Tracé Y2
        if y2_col != "(aucun)" and y2_col != y1_col and y2_col in self.df.columns:
            self.ax2.set_visible(True)
            self.ax2.plot(self.df[x_col], self.df[y2_col], label=y2_col, color='#ff7f0e', marker='x', markersize=5, linestyle='') # Correction: linestyle=''
            self.ax2.set_ylabel(y2_col.replace('_', ' ').title(), color='#ff7f0e')
            self.ax2.tick_params(axis='y', labelcolor='#ff7f0e')
            self.ax2.spines['right'].set_color('#ff7f0e')

        self.ax1.set_xlabel(x_col.replace('_', ' ').title())
        self.fig.suptitle(f"Relation entre {x_col} et les scores")
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = AnalyticsGUI(root)
    root.mainloop()