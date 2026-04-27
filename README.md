# EnglishChannelSSTSciML

> Modélisation de la température de surface de la mer (SST) dans la Manche par Scientific Machine Learning — Neural ODE sur données satellitaires.

Projet académique de 4e année d'ingénieur. L'objectif est de capturer la dynamique temporelle des SST dans la Manche à l'aide de modèles hybrides physique-ML (Neural ODE), et d'évaluer leur capacité prédictive face aux approches statistiques classiques.

---

## Contexte scientifique

La température de surface de la mer (SST) dans la Manche est un indicateur clé des dynamiques climatiques côtières. Elle présente des variations saisonnières marquées, des anomalies interannuelles et une sensibilité croissante au changement climatique. Les approches de SciML permettent d'intégrer des contraintes physiques dans un modèle appris, offrant un compromis entre interprétabilité et performance.

---

## Pipeline
Données brutes SST (satellite)
→ Extraction & préparation (Python)
→ Analyse exploratoire & statistiques (Python)
→ Pré-traitement (normalisation, détection d'anomalies)
→ Neural ODE / modèle SciML (Julia)
→ Évaluation & visualisation des prédictions

### Phases

1. **Extraction des données** — série spatio-temporelle SST sur une zone restreinte de la Manche
2. **Analyse statistique** — variations saisonnières, interannuelles, détection d'anomalies
3. **Pré-traitement** — normalisation, gestion des valeurs manquantes
4. **Modélisation SciML** — Neural ODE avec Lux.jl + SciMLSensitivity, entraînement par différentiation automatique (Zygote)
5. **Évaluation** — comparaison prédiction vs observations, métriques (RMSE, MAE), analyse de stabilité

---

## Résultats principaux

Après de multiples implémentations de modèles classifiés Scientifif Machine Learning (NODE, NSDE, NSDE latent, Koopman et SINDy), le modèle se démarquant idiosyncratiquement des autres est le modèle **SINDy** (Sparse Identification of Non-linear Dynamic system).

On notera tout d'abord que le contexte d'étude s'abroge d'un modèle baseline très basique fonctionnant très bien. En effet, en prenant la température de l'unité temporelle précédente, on s'assure une prédiction dont l'erreur est minime. **Trouver un modèle qui bat cette baseline est donc difficile** et nécessite une capture **significative** de la dynamique.

C'est ce que nous pensons avoir fait via notre implémentation de SINDy. Nous décidons de nous baser sur le Roll-out RMSE comme mesure principale de précision de nos prédictions étant donné son caractère exigeant et l'importance de pouvoir prédire une série d'anomalies de SST. Nous avons obtenu un résultat de **1.618**, que nous avons stabilisé via une **régularisation de Lyapunov** nous donnant un résultat final de **0.899**.

---

## Structure du projet
EnglishChannelSSTSciML/
├── notebooks/
│   ├── 01_data_extraction.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   └── 04_neural_ode.ipynb
├── src/                  # Scripts Julia / Python réutilisables
├── docs/                 # Figures, rapports
├── Project.toml          # Dépendances Julia
├── Manifest.toml         # Lock file Julia
└── flake.nix             # Environnement reproductible (Nix)

---

## Installation & Quick Start

### Prérequis

- Julia 1.11+ — [julialang.org](https://julialang.org/downloads/)
- Python 3.12 — via `flake.nix` ou manuellement
- (Optionnel) [Nix](https://nixos.org/) pour un environnement reproductible clé en main

### Avec Nix

```bash
git clone https://github.com/mkkuu/EnglishChannelSSTSciML
cd EnglishChannelSSTSciML
nix develop
```

### Sans Nix

```bash
# Environnement Julia
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Notebooks Python (si applicable)
pip install -r requirements.txt  # à créer
jupyter lab
```

### Données

Les données SST proviennent de [NOAA_OI_SST_V2_High_Resolution] (https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html). Elles ne sont pas incluses dans le repo pour des raisons de taille.

Pour plus de détails, consulter le notebook 00 concernant les sources de données étudiées.

---

## Dépendances

**Julia**

| Package | Rôle |
|---------|------|
| `DifferentialEquations.jl` | Résolution d'ODEs |
| `Lux.jl` | Deep learning fonctionnel |
| `SciMLSensitivity.jl` | Différentiation à travers les solveurs |
| `Zygote.jl` | Différentiation automatique |
| `Optimisers.jl` | Optimiseurs (Adam, etc.) |
| `ComponentArrays.jl` | Gestion des paramètres Neural ODE |

**Python** : 3.12.12 — utilisé pour l'extraction et l'analyse exploratoire des données.

---

## Auteurs

- **[WHenryPro](https://github.com/WHenryPro)**
- **[Mkkuu](https://github.com/mkkuu)**

Projet réalisé dans le cadre du cursus ingénieur 4e année — [Nom de l'école / cours / UE].