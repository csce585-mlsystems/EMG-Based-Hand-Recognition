import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
data = np.load('processed_data_A1.npz')
X, y = data['X'], data['y']

print(f"Dataset shape: {X.shape}")
print(f"Features per channel: {X.shape[1] // 10}")

# Split into feature types (assuming 10 channels, 6 features each)
n_channels = 10
n_features_per_channel = 6

# Extract each feature type across all channels
mav_feats = X[:, 0::n_features_per_channel]  # Every 6th starting at 0
wl_feats = X[:, 1::n_features_per_channel]   # Every 6th starting at 1
rms_feats = X[:, 2::n_features_per_channel]  # Every 6th starting at 2
zc_feats = X[:, 3::n_features_per_channel]   # Every 6th starting at 3
ssc_feats = X[:, 4::n_features_per_channel]  # Every 6th starting at 4
mf_feats = X[:, 5::n_features_per_channel]   # Every 6th starting at 5 (mean freq)

# Analyze mean frequency specifically
print("\n=== Mean Frequency Analysis ===")
print(f"Mean Freq range: [{mf_feats.min():.2f}, {mf_feats.max():.2f}]")
print(f"Mean Freq mean: {mf_feats.mean():.2f}")
print(f"Mean Freq std: {mf_feats.std():.2f}")

# Check for NaN or infinite values
print(f"NaN values in Mean Freq: {np.isnan(mf_feats).sum()}")
print(f"Inf values in Mean Freq: {np.isinf(mf_feats).sum()}")

# Compare with other features
print("\n=== Feature Statistics ===")
for name, feat in [('MAV', mav_feats), ('WL', wl_feats), ('RMS', rms_feats), 
                    ('ZC', zc_feats), ('SSC', ssc_feats), ('MF', mf_feats)]:
    print(f"{name:4s}: mean={feat.mean():8.2f}, std={feat.std():8.2f}, range=[{feat.min():8.2f}, {feat.max():8.2f}]")

# Test with and without mean frequency
print("\n=== Accuracy Comparison ===")

# Without mean frequency (first 5 features per channel)
X_no_mf = X[:, [i for i in range(X.shape[1]) if i % 6 != 5]]
print(f"Features without MF: {X_no_mf.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_no_mf, X_test_no_mf = X_no_mf[:len(X_train)], X_no_mf[len(X_train):]

# Train with all features
clf_all = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, 
                                  random_state=42, class_weight="balanced")
clf_all.fit(X_train, y_train)
acc_all = clf_all.score(X_test, y_test)

# Train without mean frequency
clf_no_mf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, 
                                    random_state=42, class_weight="balanced")
clf_no_mf.fit(X_train_no_mf, y_train)
acc_no_mf = clf_no_mf.score(X_test_no_mf, y_test)

print(f"Accuracy WITH mean frequency:    {acc_all:.4f}")
print(f"Accuracy WITHOUT mean frequency: {acc_no_mf:.4f}")
print(f"Difference: {(acc_all - acc_no_mf)*100:.2f}%")

# Feature importance analysis
importances = clf_all.feature_importances_
feature_names = []
for ch in range(10):
    for feat in ['MAV', 'WL', 'RMS', 'ZC', 'SSC', 'MF']:
        feature_names.append(f"{feat}_ch{ch}")

# Average importance by feature type
feature_types = ['MAV', 'WL', 'RMS', 'ZC', 'SSC', 'MF']
avg_importance = []
for i, ftype in enumerate(feature_types):
    indices = [j for j in range(len(importances)) if j % 6 == i]
    avg_importance.append(np.mean(importances[indices]))

print("\n=== Average Feature Importance ===")
for ftype, imp in zip(feature_types, avg_importance):
    print(f"{ftype:4s}: {imp:.4f} {'*' * int(imp * 200)}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Feature importance by type
axes[0].bar(feature_types, avg_importance, color=['#1f77b4']*5 + ['#d62728'])
axes[0].set_ylabel('Average Importance')
axes[0].set_title('Feature Importance by Type')
axes[0].grid(axis='y', alpha=0.3)

# Mean frequency distribution per gesture
for gesture in range(12):
    mask = y == gesture
    axes[1].violinplot([mf_feats[mask].flatten()], positions=[gesture], widths=0.7, 
                       showmeans=True, showmedians=False)
axes[1].set_xlabel('Gesture Class')
axes[1].set_ylabel('Mean Frequency (Hz)')
axes[1].set_title('Mean Frequency Distribution per Gesture')
axes[1].set_xticks(range(12))
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('feature_analysis.png', dpi=200)
print("\nSaved feature_analysis.png")