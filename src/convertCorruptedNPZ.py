# En Python
import numpy as np
data = np.load("data/processed/sstReducedState2COPERNICUS20102019.npz")
print(data.files)  # Devrait lister 5 clés
print(data["split"].dtype)  # Devrait être '<U10', pas 'O'