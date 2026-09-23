import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. LOAD TRAIN & TEST DATA
# =========================
train_df = pd.read_csv(r"C:\Users\hp\PycharmProjects\PythonProject3\.ipynb_checkpoints\Buche_Train_AD.csv")
test_df  = pd.read_csv(r"C:\Users\hp\PycharmProjects\PythonProject3\.ipynb_checkpoints\Buche_Test_AD.csv")

# =========================
# 2. SELECT DESCRIPTORS
# =========================
def get_X(df):
    drop_cols = [col for col in df.columns if
                 "Standardized" in col or
                 "Outlier" in col or
                 "Snew" in col or
                 "BuChE-PIC50" in col or
                 "Compound" in col]

    X = df.drop(columns=drop_cols)
    X = X.select_dtypes(include=[np.number])
    return X

X_train = get_X(train_df)
X_test  = get_X(test_df)

Y_train = train_df["BuChE-PIC50"].values
Y_test  = test_df["BuChE-PIC50"].values

X_train_mat = X_train.values
X_test_mat  = X_test.values

# =========================
# 3. BUILD MODEL (TRAIN ONLY)
# =========================
XtX_inv = np.linalg.inv(X_train_mat.T @ X_train_mat)
beta = XtX_inv @ X_train_mat.T @ Y_train

# predictions
Y_pred_train = X_train_mat @ beta
Y_pred_test  = X_test_mat @ beta

# =========================
# 4. LEVERAGE
# =========================
H_train = X_train_mat @ XtX_inv @ X_train_mat.T
h_train = np.diag(H_train)

h_test = np.array([
    x @ XtX_inv @ x.T
    for x in X_test_mat
])

# =========================
# 5. STANDARDIZED RESIDUALS
# =========================
res_train = Y_train - Y_pred_train
res_test  = Y_test - Y_pred_test

std_res_train = res_train / np.std(res_train)
std_res_test  = res_test / np.std(res_train)  # use training SD

# =========================
# 6. h* THRESHOLD
# =========================
n = X_train_mat.shape[0]
p = X_train_mat.shape[1]
n = 25
p = 3
h_star = 3*(p+1)/n

print("n =", n)
print("p =", p)
print("h* =", h_star)

# =========================
# 7. OUTLIERS
# =========================
out_train = (np.abs(std_res_train) > 3) | (h_train > h_star)
out_test  = (np.abs(std_res_test) > 3) | (h_test > h_star)

# =========================
# 8. WILLIAMS PLOT
# =========================
plt.figure(figsize=(8,6))
# =========================
# 9. PRINT OUTLIER COMPOUND IDs
# =========================

# adjust column name if needed
train_ids = train_df["Compound _ID"] if "Compound _ID" in train_df.columns else train_df.iloc[:,0]
test_ids  = test_df["Compound _ID"] if "Compound _ID" in test_df.columns else test_df.iloc[:,0]

train_outlier_ids = train_ids[out_train]
test_outlier_ids  = test_ids[out_test]

print("\nTraining set outlier compound IDs:")
print(train_outlier_ids.values)

print("\nTest set outlier compound IDs:")
print(test_outlier_ids.values)

# Training set
plt.scatter(h_train[~out_train],
            std_res_train[~out_train],
            color="blue",
            label="Training (in AD)")

plt.scatter(h_train[out_train],
            std_res_train[out_train],
            color="black",
            marker="x",
            label="Training outliers")

# Test set
plt.scatter(h_test[~out_test],
            std_res_test[~out_test],
            color="red",
            label="Test (in AD)")

plt.scatter(h_test[out_test],
            std_res_test[out_test],
            color="orange",
            marker="x",
            label="Test outliers")

# Threshold lines
plt.axhline(3, linestyle="--")
plt.axhline(-3, linestyle="--")
plt.axvline(h_star, linestyle="--")

plt.xlabel("Leverage (h)")
plt.ylabel("Standardized Residuals")
plt.title("Williams Plot BuChE (Training + Test)")
plt.legend()
plt.grid(True)

plt.show()