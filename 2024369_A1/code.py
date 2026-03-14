# importing required modules...
import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

# function to find mle mean and covariance...
def get_mean_cov(X):
    mu = np.mean(X, axis=0)
    N = X.shape[0]
    cov = np.zeros((dim, dim))
    for x in X:
        diff = (x - mu).reshape(-1, 1)
        cov += diff @ diff.T
    cov = cov / N  # MLE covariance
    return mu, cov

params = {}
for cls in [0, 1, 2]:
    X_cls = X_train_sub[y_train_sub == cls]
    mu, cov = get_mean_cov(X_cls)
    params[cls] = (mu, cov)
    print(mu.shape, cov.shape)

# Shared covariance for LDA
cov_shared = (params[0][1] + params[1][1] + params[2][1]) / 3
cov_shared += 1e-3 * np.eye(dim)
cov_shared_inv = np.linalg.inv(cov_shared)

priors = {0: 1/3, 1: 1/3, 2: 1/3}

# lda discriminant function
def lda_disc(x, mu):
    term1 = np.dot(np.dot(x, cov_shared_inv), mu)
    term2 = 0.5 * np.dot(np.dot(mu.T, cov_shared_inv), mu)
    return term1 - term2 + np.log(1/3)

# qda discriminant function
def qda_disc(x, mu, cov):
    cov_reg = cov + 1e-3 * np.eye(dim)
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

# function for per class accuracy...
def calc_per_class_accuracy(y_true, y_pred):
    for cls in [0, 1, 2]:
        mask = y_true == cls
        if np.sum(mask) > 0:
            acc = np.mean(y_pred[mask] == cls)
            print(f"Class {cls} Accuracy = {acc:.4f}")

y_pred_lda = predict_lda(X_test_sub)
y_pred_qda = predict_qda(X_test_sub)

# printing lda and qda accuracy...
print("\n=== LDA Results ===")
print("Overall LDA Accuracy =", calc_accuracy(y_test_sub, y_pred_lda))
calc_per_class_accuracy(y_test_sub, y_pred_lda)

print("\n=== QDA Results ===")
print("Overall QDA Accuracy =", calc_accuracy(y_test_sub, y_pred_qda))
calc_per_class_accuracy(y_test_sub, y_pred_qda)

# t-SNE visualization for train set
print("\nGenerating t-SNE plot for training set...")
tsne_train = TSNE(n_components=2, random_state=42, perplexity=30)
X_train_tsne = tsne_train.fit_transform(X_train_sub)

plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green']
for cls in [0, 1, 2]:
    mask = y_train_sub == cls
    plt.scatter(X_train_tsne[mask, 0], X_train_tsne[mask, 1], 
                c=colors[cls], label=f'Digit {cls}', alpha=0.6, s=50)
plt.title('t-SNE Visualization of Training Set', fontsize=14)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tsne_train.png', dpi=300)
plt.show()

# t-SNE visualization for test set
print("Generating t-SNE plot for test set...")
tsne_test = TSNE(n_components=2, random_state=42, perplexity=30)
X_test_tsne = tsne_test.fit_transform(X_test_sub)

plt.figure(figsize=(10, 8))
for cls in [0, 1, 2]:
    mask = y_test_sub == cls
    plt.scatter(X_test_tsne[mask, 0], X_test_tsne[mask, 1], 
                c=colors[cls], label=f'Digit {cls}', alpha=0.6, s=50)
plt.title('t-SNE Visualization of Test Set', fontsize=14)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tsne_test.png', dpi=300)
plt.show()

# Discriminant values for sample test points
print("\n=== Discriminant Values for Sample Test Points ===")
sample_indices = [0, 50, 150]  # one from each class

for idx in sample_indices:
    x = X_test_sub[idx]
    true_label = y_test_sub[idx]
    
    print(f"\nTest Point Index: {idx}, True Label: {true_label}")
    
    # LDA discriminant values
    print("LDA Discriminant Values:")
    lda_scores = {}
    for cls in [0, 1, 2]:
        mu, _ = params[cls]
        score = lda_disc(x, mu)
        lda_scores[cls] = score
        print(f"  Class {cls}: {score:.4f}")
    predicted_lda = max(lda_scores, key=lda_scores.get)
    print(f"  Predicted Class (LDA): {predicted_lda}")
    
    # QDA discriminant values
    print("QDA Discriminant Values:")
    qda_scores = {}
    for cls in [0, 1, 2]:
        mu, cov = params[cls]
        score = qda_disc(x, mu, cov)
        qda_scores[cls] = score
        print(f"  Class {cls}: {score:.4f}")
    predicted_qda = max(qda_scores, key=qda_scores.get)
    print(f"  Predicted Class (QDA): {predicted_qda}")