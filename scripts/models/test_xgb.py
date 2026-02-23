from xgboost import XGBClassifier
import numpy as np

X = np.random.rand(5000, 100)
y = np.random.randint(0, 2, size=(5000,))

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    tree_method="hist",
    n_jobs=-1
)

print("Fitting...")
model.fit(X, y)
print("Done.")
