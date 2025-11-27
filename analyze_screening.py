#!/usr/bin/env python3
"""
Analyse des résultats de Screening Sobol pour OCR Quality Audit.

Ce script analyse le CSV généré par le mode Screening et produit :
1. Statistiques descriptives
2. Effets principaux de chaque paramètre
3. Matrice de corrélation
4. Visualisations (graphiques)
5. Rapport texte détaillé

Usage:
    python analyze_screening.py screening_sobol_9_20250127_143052.csv
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

def load_screening_data(csv_file):
    """Charge le CSV de screening."""
    print(f"📂 Chargement de {csv_file}...")
    df = pd.read_csv(csv_file, sep=';')
    print(f"✅ {len(df)} points chargés")
    return df

def compute_statistics(df, target='score_tesseract'):
    """Calcule les statistiques descriptives."""
    print(f"\n📊 === STATISTIQUES DESCRIPTIVES ({target}) ===")
    print(f"Moyenne : {df[target].mean():.2f}%")
    print(f"Écart-type : {df[target].std():.2f}%")
    print(f"Min : {df[target].min():.2f}%")
    print(f"Max : {df[target].max():.2f}%")
    print(f"Médiane : {df[target].median():.2f}%")

    # Top 5
    print(f"\n🏆 TOP 5 Meilleurs scores :")
    top5 = df.nlargest(5, target)
    for idx, row in top5.iterrows():
        print(f"  Point {row['point_id']}: {row[target]:.2f}%")

    return {
        'mean': df[target].mean(),
        'std': df[target].std(),
        'min': df[target].min(),
        'max': df[target].max(),
        'top5': top5
    }

def compute_main_effects(df, param_cols, target='score_tesseract', n_bins=10):
    """
    Calcule les effets principaux de chaque paramètre.

    Méthode : Divise chaque paramètre en n_bins, calcule la moyenne
    du score dans chaque bin, puis mesure la variation.
    """
    print(f"\n🔍 === EFFETS PRINCIPAUX (Impact sur {target}) ===")

    effects = {}

    for param in param_cols:
        # Diviser en bins
        df['bin'] = pd.cut(df[param], bins=n_bins, labels=False)
        bin_means = df.groupby('bin')[target].mean()

        # Effet = écart-type des moyennes par bin
        effect = bin_means.std()

        # Amplitude min-max
        amplitude = bin_means.max() - bin_means.min()

        effects[param] = {
            'effect_std': effect,
            'amplitude': amplitude,
            'bin_means': bin_means
        }

        print(f"{param:20s} | Effet: {effect:6.2f} | Amplitude: {amplitude:6.2f}%")

    # Tri par effet décroissant
    sorted_effects = sorted(effects.items(), key=lambda x: x[1]['effect_std'], reverse=True)

    print(f"\n📈 Classement par impact (du plus au moins influent) :")
    for i, (param, data) in enumerate(sorted_effects, 1):
        print(f"  {i}. {param:20s} (effet: {data['effect_std']:.2f})")

    return effects, sorted_effects

def compute_correlations(df, param_cols, target='score_tesseract'):
    """Calcule la matrice de corrélation."""
    print(f"\n🔗 === CORRÉLATIONS AVEC {target} ===")

    # Corrélations avec le target
    correlations = df[param_cols + [target]].corr()[target].drop(target).sort_values(ascending=False)

    print("Corrélations (Pearson) :")
    for param, corr in correlations.items():
        symbol = "📈" if abs(corr) > 0.3 else "  "
        print(f"  {symbol} {param:20s}: {corr:+.3f}")

    # Corrélations entre paramètres
    param_corr = df[param_cols].corr()

    print(f"\n⚠️ Corrélations fortes entre paramètres (|r| > 0.5) :")
    found_strong = False
    for i, param1 in enumerate(param_cols):
        for param2 in param_cols[i+1:]:
            corr_val = param_corr.loc[param1, param2]
            if abs(corr_val) > 0.5:
                print(f"  {param1} ↔ {param2}: {corr_val:+.3f}")
                found_strong = True

    if not found_strong:
        print("  Aucune corrélation forte détectée (bon signe !)")

    return correlations, param_corr

def plot_main_effects(effects, sorted_effects, target='score_tesseract', output_dir='analysis_plots'):
    """Génère les graphiques des effets principaux."""
    Path(output_dir).mkdir(exist_ok=True)

    print(f"\n📊 Génération des graphiques...")

    # 1. Barplot des effets
    fig, ax = plt.subplots(figsize=(10, 6))
    params = [p for p, _ in sorted_effects]
    effect_values = [data['effect_std'] for _, data in sorted_effects]

    bars = ax.barh(params, effect_values, color='steelblue')
    ax.set_xlabel('Effet (std des moyennes par bin)')
    ax.set_title(f'Effets Principaux sur {target}')
    ax.grid(axis='x', alpha=0.3)

    # Colorer les 3 plus influents
    for i in range(min(3, len(bars))):
        bars[-(i+1)].set_color('coral')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/main_effects.png', dpi=150)
    print(f"  ✅ {output_dir}/main_effects.png")

    # 2. Graphiques individuels des top 4 paramètres
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (param, data) in enumerate(sorted_effects[:4]):
        ax = axes[i]
        bin_means = data['bin_means']
        ax.plot(bin_means.index, bin_means.values, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel(f'{param} (bin)')
        ax.set_ylabel(f'{target} (%)')
        ax.set_title(f'Effet de {param}')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/top4_effects_detail.png', dpi=150)
    print(f"  ✅ {output_dir}/top4_effects_detail.png")

    plt.close('all')

def plot_correlations(correlations, param_corr, output_dir='analysis_plots'):
    """Génère les graphiques de corrélation."""

    # 1. Heatmap corrélations avec target
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_df = correlations.to_frame(name='Correlation').sort_values('Correlation', ascending=False)

    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                cbar_kws={'label': 'Corrélation'}, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Corrélations avec score_tesseract')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlations_target.png', dpi=150)
    print(f"  ✅ {output_dir}/correlations_target.png")

    # 2. Heatmap corrélations entre paramètres
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(param_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                cbar_kws={'label': 'Corrélation'}, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Corrélations entre paramètres')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlations_params.png', dpi=150)
    print(f"  ✅ {output_dir}/correlations_params.png")

    plt.close('all')

def plot_score_distribution(df, target='score_tesseract', output_dir='analysis_plots'):
    """Génère l'histogramme de distribution des scores."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df[target], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(df[target].mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {df[target].mean():.2f}%')
    ax.axvline(df[target].median(), color='orange', linestyle='--', linewidth=2, label=f'Médiane: {df[target].median():.2f}%')

    ax.set_xlabel(f'{target} (%)')
    ax.set_ylabel('Fréquence')
    ax.set_title(f'Distribution de {target}')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/score_distribution.png', dpi=150)
    print(f"  ✅ {output_dir}/score_distribution.png")

    plt.close('all')

def generate_report(csv_file, stats, sorted_effects, correlations, output_dir='analysis_plots'):
    """Génère un rapport texte détaillé."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{output_dir}/rapport_analyse_{timestamp}.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RAPPORT D'ANALYSE - SCREENING SOBOL\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Fichier analysé : {csv_file}\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("=" * 80 + "\n")
        f.write("1. STATISTIQUES DESCRIPTIVES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Moyenne : {stats['mean']:.2f}%\n")
        f.write(f"Écart-type : {stats['std']:.2f}%\n")
        f.write(f"Min : {stats['min']:.2f}%\n")
        f.write(f"Max : {stats['max']:.2f}%\n")
        f.write(f"Amplitude : {stats['max'] - stats['min']:.2f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("2. CLASSEMENT DES PARAMÈTRES PAR INFLUENCE\n")
        f.write("=" * 80 + "\n")
        for i, (param, data) in enumerate(sorted_effects, 1):
            f.write(f"{i}. {param:20s} - Effet: {data['effect_std']:6.2f} - Amplitude: {data['amplitude']:6.2f}%\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("3. CORRÉLATIONS AVEC LE SCORE TESSERACT\n")
        f.write("=" * 80 + "\n")
        for param, corr in correlations.items():
            f.write(f"{param:20s}: {corr:+.3f}\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("4. RECOMMANDATIONS\n")
        f.write("=" * 80 + "\n\n")

        # Top 3 paramètres influents
        top3 = sorted_effects[:3]
        f.write("Paramètres à OPTIMISER en priorité :\n")
        for i, (param, data) in enumerate(top3, 1):
            f.write(f"  {i}. {param} (effet: {data['effect_std']:.2f})\n")

        f.write("\n")

        # Paramètres peu influents
        if len(sorted_effects) > 5:
            bottom3 = sorted_effects[-3:]
            f.write("Paramètres pouvant être FIXÉS (peu d'impact) :\n")
            for param, data in bottom3:
                f.write(f"  - {param} (effet: {data['effect_std']:.2f})\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("FIN DU RAPPORT\n")
        f.write("=" * 80 + "\n")

    print(f"  ✅ {report_file}")
    return report_file

def main():
    if len(sys.argv) < 2:
        print("❌ Usage: python analyze_screening.py <fichier_csv>")
        print("   Exemple: python analyze_screening.py screening_sobol_9_20250127_143052.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    if not Path(csv_file).exists():
        print(f"❌ Erreur: Fichier {csv_file} non trouvé")
        sys.exit(1)

    print("=" * 80)
    print("🔬 ANALYSE DES RÉSULTATS DE SCREENING SOBOL")
    print("=" * 80)

    # Charger les données
    df = load_screening_data(csv_file)

    # Identifier les colonnes de paramètres
    score_cols = ['point_id', 'score_tesseract', 'score_nettete', 'score_contraste']
    param_cols = [col for col in df.columns if col not in score_cols]

    print(f"\n📋 Paramètres analysés : {', '.join(param_cols)}")

    # Statistiques
    stats = compute_statistics(df)

    # Effets principaux
    effects, sorted_effects = compute_main_effects(df, param_cols)

    # Corrélations
    correlations, param_corr = compute_correlations(df, param_cols)

    # Graphiques
    output_dir = f"analysis_{Path(csv_file).stem}"
    plot_main_effects(effects, sorted_effects, output_dir=output_dir)
    plot_correlations(correlations, param_corr, output_dir=output_dir)
    plot_score_distribution(df, output_dir=output_dir)

    # Rapport
    report_file = generate_report(csv_file, stats, sorted_effects, correlations, output_dir=output_dir)

    print("\n" + "=" * 80)
    print("✅ ANALYSE TERMINÉE")
    print("=" * 80)
    print(f"\n📁 Résultats sauvegardés dans : {output_dir}/")
    print(f"📄 Rapport texte : {report_file}")
    print(f"\n💡 Consultez les graphiques pour visualiser les résultats !")

if __name__ == "__main__":
    main()
