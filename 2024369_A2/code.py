# importing required modules...
import numpy as np
import struct
import matplotlib.pyplot as plt

# Loading training images...
with open("train-images.idx3-ubyte", 'rb') as f:
    magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8)
    train_imgs = train_imgs.reshape(count, rows, cols)

# Loading training labels...
with open("train-labels.idx1-ubyte", 'rb') as f:
    magic, count = struct.unpack(">II", f.read(8))
    train_lbls = np.frombuffer(f.read(), dtype=np.uint8)

# Loading test images...
with open("t10k-images.idx3-ubyte", 'rb') as f:
    magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8)
    test_imgs = test_imgs.reshape(count, rows, cols)

# Loading test labels...
with open("t10k-labels.idx1-ubyte", 'rb') as f:
    magic, count = struct.unpack(">II", f.read(8))
    test_lbls = np.frombuffer(f.read(), dtype=np.uint8)

# printing just to check if we are doing right...
print(train_imgs.shape, train_lbls.shape)
print(test_imgs.shape, test_lbls.shape)

# filtering digits 0,1,2 as per assignment...
train_mask = np.isin(train_lbls, [0, 1, 2])
test_mask = np.isin(test_lbls, [0, 1, 2])

X_train = train_imgs[train_mask]
y_train = train_lbls[train_mask]

X_test = test_imgs[test_mask]
y_test = test_lbls[test_mask]

# function to random sampling (100 per class)...
def sample_data(X, y, n=100):
    X_list = []
    y_list = []
    for cls in [0, 1, 2]:
        idx = np.where(y == cls)[0]
        chosen = np.random.choice(idx, n, replace=False)
        X_list.append(X[chosen])
        y_list.append(y[chosen])
    return np.vstack(X_list), np.hstack(y_list)

X_train_sub, y_train_sub = sample_data(X_train, y_train, 100)
X_test_sub, y_test_sub = sample_data(X_test, y_test, 100)

# flatten and normalize...
X_train_sub = X_train_sub.reshape(300, -1) / 255.0
X_test_sub = X_test_sub.reshape(300, -1) / 255.0

# Defining dimension
dim = X_train_sub.shape[1]

def fit_pca(X_row, variance_threshold=0.95):
    N, _ = X_row.shape
    X = X_row.T                          # (784, 300)
    mu = X.mean(axis=1, keepdims=True)   # (784, 1)
    Xc = X - mu                          # (784, 300)

    S = np.dot(Xc, Xc.T) / (N - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(S)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = eigenvalues.sum()
    cumvar = np.cumsum(eigenvalues) / total_var
    k = int(np.searchsorted(cumvar, variance_threshold)) + 1

    Up = eigenvectors[:, :k]             # (784, k)
    return Up, mu, eigenvalues, cumvar, k

def transform_pca(X_row, Up, mu):
    X = X_row.T                          # (D, N)
    Xc = X - mu
    return np.dot(Up.T,Xc)                     # (k, N)

def reconstruct_pca(Y, Up, mu):
    return (np.dot(Up,Y) + mu).T

# Fit PCA with 75% variance
Up_75, mu_train, eigvals, cumvar, k_75 = fit_pca(X_train_sub, variance_threshold=0.75)
print(f"\nPCA 75% variance -> k = {k_75} components")

Y_train_pca75 = transform_pca(X_train_sub, Up_75, mu_train)   # (k, 300)
Y_test_pca75  = transform_pca(X_test_sub,  Up_75, mu_train)

# Reconstruction + MSE for 5 samples
n_show  = 5
X_recon = reconstruct_pca(Y_train_pca75[:, :n_show], Up_75, mu_train)

fig, axes = plt.subplots(2, n_show, figsize=(12, 5))
fig.suptitle("PCA Reconstruction (75% variance) - First 5 Training Samples", fontsize=13)
for i in range(n_show):
    mse = np.mean((X_train_sub[i] - X_recon[i]) ** 2)
    axes[0, i].imshow(X_train_sub[i].reshape(28, 28), cmap='gray')
    axes[0, i].set_title(f"Original\nlabel={y_train_sub[i]}", fontsize=8)
    axes[0, i].axis('off')
    axes[1, i].imshow(X_recon[i].reshape(28, 28), cmap='gray')
    axes[1, i].set_title(f"Recon\nMSE={mse:.4f}", fontsize=8)
    axes[1, i].axis('off')
plt.tight_layout()
plt.savefig("pca_reconstruction.png", dpi=120)
plt.show()

def fit_fda(X_row, y):
    classes = np.unique(y)
    N, D = X_row.shape
    mu_all = X_row.mean(axis=0)

    # Between-class scatter
    SB = np.zeros((D, D))
    for c in classes:
        Xc_data = X_row[y == c]
        Nc = Xc_data.shape[0]
        mu_c = Xc_data.mean(axis=0)
        diff = (mu_c - mu_all).reshape(-1, 1)
        SB += Nc * (np.dot(diff,diff.T))

    # Within-class scatter
    SW = np.zeros((D, D))
    for c in classes:
        Xc_data = X_row[y == c]
        mu_c = Xc_data.mean(axis=0)
        diff = Xc_data - mu_c
        SW += np.dot(diff.T,diff)

    # Small regularisation on SW for numerical stability
    SW_reg = SW + 1e-6 * np.eye(D)
    # Convert generalized problem SB w = lambda SW w
    # to standard form: SW_inv @ SB w = lambda w
    SW_inv = np.linalg.pinv(SW_reg)
    eigenvalues, eigenvectors = np.linalg.eigh(SW_inv @ SB)  # standard eigenvalue problem

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    W = eigenvectors[:, :len(classes) - 1]
    return W, SB, SW

def transform_fda(X_row, W):
    return np.dot(X_row,W)

W_fda, SB, SW = fit_fda(X_train_sub, y_train_sub)
print(f"\nFDA projection matrix W shape: {W_fda.shape}")

Z_train_fda = transform_fda(X_train_sub, W_fda)   # (300, 2)
Z_test_fda  = transform_fda(X_test_sub,  W_fda)   # (300, 2)

# function to find mle mean and covariance...
def get_mean_cov(X):
    mu = np.mean(X, axis=0)
    N = X.shape[0]
    d = X.shape[1]
    cov = np.zeros((d, d))
    for x in X:
        diff = (x - mu).reshape(-1, 1)
        cov += np.dot(diff,diff.T)
    cov = cov / N  # MLE covariance
    return mu, cov

def build_params(X, y):
    params = {}
    for cls in [0, 1, 2]:
        X_cls = X[y == cls]
        mu, cov = get_mean_cov(X_cls)
        params[cls] = (mu, cov)

    d = X.shape[1]
    cov_shared = (params[0][1] + params[1][1] + params[2][1]) / 3
    cov_shared += 1e-3 * np.eye(d)
    cov_shared_inv = np.linalg.inv(cov_shared)
    return params, cov_shared_inv

# lda discriminant function
def lda_disc(x, mu):
    term1 = np.dot(np.dot(x, cov_shared_inv), mu)
    term2 = 0.5 * np.dot(np.dot(mu.T, cov_shared_inv), mu)
    return term1 - term2 + np.log(1/3)

# qda discriminant function
def qda_disc(x, mu, cov):
    d = len(mu)
    cov_reg = cov + 1e-3 * np.eye(d)
    try:
        cov_inv = np.linalg.inv(cov_reg)
        det = np.linalg.det(cov_reg)
        if det <= 0:
            det = 1e-10
        diff = x - mu
        term1 = -0.5 * np.log(det)
        term2 = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
        return term1 + term2 + np.log(1/3)
    except np.linalg.LinAlgError:
        return -np.inf

# lda prediction function
def predict_lda(X):
    preds = []
    for x in X:
        scores = {}
        for cls in [0, 1, 2]:
            mu, _ = params[cls]
            scores[cls] = lda_disc(x, mu)
        preds.append(max(scores, key=scores.get))
    return np.array(preds)

# qda prediction function
def predict_qda(X):
    preds = []
    for x in X:
        scores = {}
        for cls in [0, 1, 2]:
            mu, cov = params[cls]
            scores[cls] = qda_disc(x, mu, cov)
        preds.append(max(scores, key=scores.get))
    return np.array(preds)

# function for finding accuracy score...
def calc_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

results = {}

# (A) FDA -> LDA
params, cov_shared_inv = build_params(Z_train_fda, y_train_sub)
results["FDA+LDA"] = {
    "train": calc_accuracy(y_train_sub, predict_lda(Z_train_fda)),
    "test":  calc_accuracy(y_test_sub,  predict_lda(Z_test_fda)),
}

# (B) FDA -> QDA (reuses same params from above)
results["FDA+QDA"] = {
    "train": calc_accuracy(y_train_sub, predict_qda(Z_train_fda)),
    "test":  calc_accuracy(y_test_sub,  predict_qda(Z_test_fda)),
}

# (C) PCA (75%) -> LDA
params, cov_shared_inv = build_params(Y_train_pca75.T, y_train_sub)
results["PCA75%+LDA"] = {
    "train": calc_accuracy(y_train_sub, predict_lda(Y_train_pca75.T)),
    "test":  calc_accuracy(y_test_sub,  predict_lda(Y_test_pca75.T)),
}

# (D) PCA (90%) -> LDA
Up_90, _, _, _, k_90 = fit_pca(X_train_sub, variance_threshold=0.90)
Y_train_pca90 = transform_pca(X_train_sub, Up_90, mu_train)
Y_test_pca90  = transform_pca(X_test_sub,  Up_90, mu_train)
print(f"PCA 90% variance -> k = {k_90} components")
params, cov_shared_inv = build_params(Y_train_pca90.T, y_train_sub)
results["PCA90%+LDA"] = {
    "train": calc_accuracy(y_train_sub, predict_lda(Y_train_pca90.T)),
    "test":  calc_accuracy(y_test_sub,  predict_lda(Y_test_pca90.T)),
}

# (E) PCA first-2 components -> LDA
Up_all, _, _, _, _ = fit_pca(X_train_sub, variance_threshold=0.9999)
Up_2 = Up_all[:, :2]
Y_train_pca2 = transform_pca(X_train_sub, Up_2, mu_train)
Y_test_pca2  = transform_pca(X_test_sub,  Up_2, mu_train)
params, cov_shared_inv = build_params(Y_train_pca2.T, y_train_sub)
results["PCA-2PC+LDA"] = {
    "train": calc_accuracy(y_train_sub, predict_lda(Y_train_pca2.T)),
    "test":  calc_accuracy(y_test_sub,  predict_lda(Y_test_pca2.T)),
}

# Print summary table
print("\n" + "="*55)
print(f"{'Method':<20} {'Train Acc':>10} {'Test Acc':>10}")
print("="*55)
for name, acc in results.items():
    print(f"{name:<20} {acc['train']*100:>9.2f}% {acc['test']*100:>9.2f}%")
print("="*55)

# plotting the 2d projections for both FDA and PCA...
clr = {0: 'red', 1: 'green', 2: 'blue'}
mrk = {0: 'o', 1: 's', 2: '^'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("2D projections - FDA vs PCA", fontsize=13)

# FDA plot
ax = axes[0]
for c in [0, 1, 2]:
    idx = y_train_sub == c
    ax.scatter(Z_train_fda[idx, 0], Z_train_fda[idx, 1],
               color=clr[c], marker=mrk[c], label=f'class {c}', alpha=0.6, s=40)
ax.set_title("FDA (2 directions)")
ax.set_xlabel("direction 1")
ax.set_ylabel("direction 2")
ax.legend()
ax.grid(True)

# PCA plot (first 2 PCs)
ax = axes[1]
for c in [0, 1, 2]:
    idx = y_train_sub == c
    ax.scatter(Y_train_pca2[0, idx], Y_train_pca2[1, idx],
               color=clr[c], marker=mrk[c], label=f'class {c}', alpha=0.6, s=40)
ax.set_title("PCA (first 2 components)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig("feature_space_2d.png", dpi=120)
plt.show()

# cumulative variance explained by PCA...
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(1, len(cumvar)+1), cumvar * 100, color='blue')
ax.axhline(75, linestyle='--', color='orange', label='75%')
ax.axhline(90, linestyle='--', color='red', label='90%')
ax.axvline(k_75, linestyle=':', color='orange')
ax.axvline(k_90, linestyle=':', color='red')
ax.set_xlabel("no. of components")
ax.set_ylabel("variance explained (%)")
ax.set_title("PCA variance plot")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("pca_variance.png", dpi=120)
plt.show()