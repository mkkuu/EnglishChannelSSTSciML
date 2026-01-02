# ============================================================
# 0. Imports
# ============================================================

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ============================================================
# 1. Charger les données Copernicus (NetCDF)
# ============================================================

ds = xr.open_dataset("sst_manche_2010_2020.nc")
print(ds)

# Variable principale
sst = ds["sst"]

# Conversion en °C si nécessaire
if sst.attrs.get("units", "").lower() in ["kelvin", "k"]:
    sst = sst - 273.15
    sst.attrs["units"] = "degC"

# ============================================================
# 2. Vérifications rapides
# ============================================================

print("Période :", sst.time.values[0], "→", sst.time.values[-1])
print("Latitude :", float(sst.latitude.min()), "→", float(sst.latitude.max()))
print("Longitude :", float(sst.longitude.min()), "→", float(sst.longitude.max()))

n_lat = sst.sizes["latitude"]
n_lon = sst.sizes["longitude"]
print("Points spatiaux par jour :", n_lat * n_lon)

# ============================================================
# 3. Préparation des données pour PCA
# ============================================================

# Aplatir l'espace
sst_flat = sst.stack(space=("latitude", "longitude"))

# Retirer la moyenne temporelle (anomalies)
sst_anom = sst_flat - sst_flat.mean(dim="time")

# Conversion en numpy
X = sst_anom.values  # shape = (temps, espace)

# Gestion des NaN (sécurité)
X = np.nan_to_num(X)

# ============================================================
# 4. PCA
# ============================================================

n_components = 3
pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X)

print("Variance expliquée par composante :")
for i, v in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {100*v:.2f} %")

print("Variance cumulée :", 100 * pca.explained_variance_ratio_.sum(), "%")

# ============================================================
# 5. Visualisation des composantes temporelles
# ============================================================

time = sst.time.values

plt.figure(figsize=(10, 4))
for i in range(n_components):
    plt.plot(time, X_pca[:, i], label=f"PC{i+1}")

plt.legend()
plt.title("Composantes principales temporelles (SST Manche)")
plt.xlabel("Temps")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# ============================================================
# 6. Reconstruction (optionnelle, pour validation)
# ============================================================

X_rec = pca.inverse_transform(X_pca)
reconstruction_error = np.mean((X - X_rec)**2)

print("Erreur quadratique moyenne de reconstruction :", reconstruction_error)

# ============================================================
# 7. État dynamique final pour SciML
# ============================================================

# État réduit x(t)
x_t = X_pca  # shape = (temps, n_components)

print("État dynamique x(t) :", x_t.shape)

# Sauvegarde pour la suite
np.save("etat_dynamique_PCA.npy", x_t)

print("Pipeline terminé avec succès.")
