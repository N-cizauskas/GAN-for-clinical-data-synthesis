# import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import ks_2samp, chi2_contingency
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from scipy.stats import fisher_exact


# paths
DATA_PATH = r"C:/Users/c3058452/OneDrive - Newcastle University/Work in Progress/Saved_Rdata/testdata2.csv"
OUT_DIR = r"C:/Users/c3058452/OneDrive - Newcastle University/Work in Progress/GAN"
os.makedirs(OUT_DIR, exist_ok=True)

# hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 32                 # larger latent space
cond_dim = 1                    # conditioning on Treat (0/1) as single scalar
batch_size = 512
lr_g = 1e-4
lr_d = 1e-4
num_epochs = 1000
beta1, beta2 = 0.5, 0.999       # Adam betas commonly used in GANs
label_smooth = 0.9              # real label smoothing to 0.9
patience = 50                   # not used for early stopping here, train fixed epochs
print_every = 1

# data
df = pd.read_csv(DATA_PATH)
# features 
feature_names = ['Sex', 'Outcome', 'Age', "Ethnicity", "Age_diagnosis", "Location", "Years_treatment"]
assert all(col in df.columns for col in feature_names), f"Missing columns in CSV: {feature_names}"

# separate features and condition (treat)
X = df[feature_names].copy()
cond = df['Treat'].values.reshape(-1, 1)  # condition for cGAN

# identify binary vs continuous
binary_idx = [0, 1]  # Sex, Outcome
categorical_cols = ['Ethnicity', 'Location']  # need one-hot encoding
cont_idx = [2, 4, 6]  # Age, Age_diagnosis, Years_treatment

# different names: 
binary_features = ['Sex', 'Outcome']                # binary 0/1 features
continuous_features = ['Age', 'Age_diagnosis', 'Years_treatment']
categorical_features = ['Ethnicity', 'Location']    # multi-category variables

# one-hot encode categorical columns
X_cat = pd.get_dummies(X[categorical_cols], drop_first=True).values

# scale continuous features
# define and fit scaler
scaler = StandardScaler()
X_cont = scaler.fit_transform(X.iloc[:, cont_idx])

# reassemble: [binary | categorical | continuous]
X_proc = np.hstack([X.iloc[:, binary_idx].astype(float).values, X_cat, X_cont])

n_binary = len(binary_idx)
n_categorical = X_cat.shape[1]
n_cont = X_cont.shape[1]
n_features = n_binary + n_categorical + n_cont


# convert to tensors
features_tensor = torch.tensor(X_proc, dtype=torch.float32).to(device)
cond_tensor = torch.tensor(cond, dtype=torch.float32).to(device)

# create DataLoader
dataset = TensorDataset(features_tensor, cond_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

n_features = X_proc.shape[1]




# GAN model
def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.005)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, latent_dim, cond_dim, output_dim):
        super().__init__()
        input_dim = latent_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 512),       # new extra layer
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, output_dim)
)
            # final activations handled in forward to separate binary & continuous
        
    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        out = self.net(x)
        # out shape: (batch, n_features) -> split to binaries & continuous
        # apply sigmoid to binary dims (first len(binary_idx)), keep continuous linear
        if len(binary_idx) > 0:
            n_bin_cat = n_binary + n_categorical
            bin_out = torch.sigmoid(out[:, :n_bin_cat])
            cont_out = out[:, n_bin_cat:]
            return torch.cat([bin_out, cont_out], dim=1)
        else:
            return out

class Discriminator(nn.Module):
    def __init__(self, input_dim, cond_dim):
        super().__init__()
        # discriminator receives features concatenated with condition
        in_dim = input_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)  #  no sigmoid here
        )
    def forward(self, x, c):
        xc = torch.cat([x, c], dim=1)
        return self.net(xc).squeeze(1)

# instantiate
G = Generator(latent_dim=latent_dim, cond_dim=cond_dim, output_dim=n_features).to(device)
D = Discriminator(input_dim=n_features, cond_dim=cond_dim).to(device)
G.apply(weights_init)
D.apply(weights_init)

# loss optimization 
criterion = nn.BCEWithLogitsLoss()  # stable combined sigmoid+BCELoss
optimizer_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=1e-5, betas=(0.5, 0.999))

# training loop
G.train(); D.train()
g_losses = []
d_losses = []

# early stopping setup
best_g_loss = float("inf")
patience = 50     # number of epochs with no improvement allowed
patience_counter = 0

for epoch in range(1, num_epochs + 1):
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0

    for real_x, real_c in train_loader:
        bs = real_x.size(0)

        # train discriminator
        optimizer_D.zero_grad()

        # real labels 
        real_labels = torch.full((bs,), 0.9, device=device)

        #real_labels = torch.full((bs,), label_smooth, device=device)
        #real_labels = real_labels - 0.05 * torch.rand_like(real_labels)

        # fake labels
        fake_labels = torch.zeros(bs, device=device)

        # real batch
        real_logits = D(real_x, real_c).view(-1)
        loss_real = criterion(real_logits, real_labels)

        # fake batch
        z = torch.randn(bs, latent_dim, device=device)
        fake_x = G(z, real_c).detach()
        fake_logits = D(fake_x, real_c).view(-1)
        loss_fake = criterion(fake_logits, fake_labels)

        # combine losses
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        # train generator
        optimizer_G.zero_grad()
        z = torch.randn(bs, latent_dim, device=device)
        gen_x = G(z, real_c)
        gen_logits = D(gen_x, real_c).view(-1)

        target_for_g = torch.full((bs,), label_smooth, device=device)
        loss_G = criterion(gen_logits, target_for_g)
        loss_G.backward()
        optimizer_G.step()

        # accumulate losses
        epoch_d_loss += loss_D.item()
        epoch_g_loss += loss_G.item()

    # end - average losses
    epoch_d_loss /= len(train_loader)
    epoch_g_loss /= len(train_loader)
    d_losses.append(epoch_d_loss)
    g_losses.append(epoch_g_loss)

    
    # early stopping check
    if epoch_g_loss < best_g_loss - 1e-4:  # require slight improvement
        best_g_loss = epoch_g_loss
        patience_counter = 0
        torch.save(G.state_dict(), os.path.join(OUT_DIR, "best_generator.pth"))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}, generator loss stopped improving.")
            break

    # average losses for epoch
    epoch_d_loss /= len(train_loader)
    epoch_g_loss /= len(train_loader)
    d_losses.append(epoch_d_loss)
    g_losses.append(epoch_g_loss)

    if epoch % print_every == 0:
        print(f"Epoch {epoch}/{num_epochs}  |  Loss D: {epoch_d_loss:.4f}  |  Loss G: {epoch_g_loss:.4f}")

# save models
torch.save(G.state_dict(), os.path.join(OUT_DIR, "generator_cgan.pth"))
torch.save(D.state_dict(), os.path.join(OUT_DIR, "discriminator_cgan.pth"))

## generate the synthetic controls
G.eval()
n_gen = min(1000, len(df))  # number of synthetic samples
z = torch.randn(n_gen, latent_dim, device=device)
cond_zero = torch.zeros(n_gen, cond_dim, device=device)  # condition = 0 for controls
with torch.no_grad():
    synth = G(z, cond_zero).cpu().numpy()

# post-process synthetic outputs:
# first binary dims are in [0,1] -> threshold at 0.5 to get integer categories
n_bin_cat = len(binary_features) + len(pd.get_dummies(df[categorical_features], drop_first=True).columns)

synth_bin_cat = (synth[:, :n_bin_cat] >= 0.5).astype(int)
synth_cont = synth[:, n_bin_cat:]
synth_cont_unscaled = scaler.inverse_transform(synth_cont)
# inverse transform continuous features 
cont_feature_names = ['Age', 'Age_diagnosis', 'Years_treatment']
if len(cont_idx) > 0:
    synth_cont_unscaled = pd.DataFrame(scaler.inverse_transform(synth_cont),
                                       columns=cont_feature_names)
else:
    synth_cont_unscaled = pd.DataFrame(np.empty((n_gen, 0)))


cat_encoded_cols = list(pd.get_dummies(X[categorical_cols], drop_first=True).columns)
all_feature_cols = (
    [X.columns[i] for i in binary_idx] +
    cat_encoded_cols +
    [X.columns[i] for i in cont_idx]
)

synth_full = np.hstack([synth_bin_cat, synth_cont_unscaled.values])
synthetic_df = pd.DataFrame(synth_full, columns=feature_names)
synthetic_df['Treat'] = 0

# make sure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# save synthetic data safely
output_path = os.path.join(OUT_DIR, "synthetic_controls.csv")
synthetic_df.to_csv(output_path, index=False)

print("Synthetic data saved to:", output_path)

# comparisons
real_controls = df[df['Treat'] == 0].reset_index(drop=True)
if len(real_controls) == 0:
    # if no real controls exist, compare to whole dataset
    print("No real controls found; comparing synthetic controls to whole dataset.")
    real_compare = real_controls[feature_names] 
else:
    real_compare = real_controls[feature_names]

# for continuous use KS test
if len(cont_idx) > 0:
    cont_col = feature_names[cont_idx[0]]  # get column name
    age_real = real_compare[cont_col].values  # extract as NumPy array
    age_synth = synthetic_df[cont_col].values
    ks_stat, ks_p = ks_2samp(age_real, age_synth)
    print(f"KS test for Age: stat={ks_stat:.4f}, p={ks_p:.4f}")
else:
    print("No continuous feature to run KS test.")

# cont features KS test
if continuous_features:
    for feature in continuous_features:
        real_vals = real_compare[feature].values
        synth_vals = synthetic_df[feature].values
        ks_stat, ks_p = ks_2samp(real_vals, synth_vals)
        print(f"KS test for {feature}: stat={ks_stat:.4f}, p={ks_p:.4f}")
else:
    print("No continuous features to run KS test.")


# combine binary and cat features
binary_and_cat_features = binary_features + categorical_features

for feature in binary_and_cat_features:
    if feature not in real_compare.columns or feature not in synthetic_df.columns:
        print(f"Skipping {feature}: column missing in real or synthetic data.")
        continue

    # define number of cat
    n_categories = int(max(real_compare[feature].max(), synthetic_df[feature].max())) + 1

    # clip values to valid indices
    real_vals = np.clip(real_compare[feature].astype(int).values, 0, n_categories - 1)
    synth_vals = np.clip(synthetic_df[feature].astype(int).values, 0, n_categories - 1)

    # skip feature if zero variance
    if len(np.unique(real_vals)) == 1 and len(np.unique(synth_vals)) == 1:
        print(f"Skipping {feature}: zero variance in both real and synthetic data")
        continue

    # get counts
    real_counts = np.bincount(real_vals, minlength=n_categories)
    synth_counts = np.bincount(synth_vals, minlength=n_categories)
    contingency = np.vstack([real_counts, synth_counts])

    try:
        if n_categories == 2:
            # fisher's exact for binary features
            oddsratio, p = fisher_exact(contingency)
            print(f"Fisher's exact test for {feature}: p={p:.4f}, "
                  f"real_counts={real_counts}, synth_counts={synth_counts}")
        else:
            # chi square for multi category features
            chi2, p, dof, expected = chi2_contingency(contingency)
            print(f"Chi-square for {feature}: chi2={chi2:.3f}, p={p:.4f}, "
                  f"real_counts={real_counts}, synth_counts={synth_counts}")
    except ValueError as e:
        print(f"Skipping {feature} due to error in contingency test: {e}")





# graphs
binary_cols = ['Sex', 'Outcome']
categorical_cols = ['Ethnicity', 'Location']
cont_cols = ['Age', 'Age_diagnosis', 'Years_treatment']
feature_names = binary_cols + categorical_cols + cont_cols
cat_encoded_cols = list(pd.get_dummies(df[categorical_cols], drop_first=True).columns)
processed_feature_names = binary_cols + cat_encoded_cols + cont_cols



# loss curves
plt.figure(figsize=(8,5))
plt.plot(d_losses, label="Discriminator Loss")
plt.plot(g_losses, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_curves.png"), dpi=200)
plt.close()


# def feature groups
binary_cols = ['Sex', 'Outcome']
categorical_cols = ['Ethnicity', 'Location']
cont_cols = ['Age', 'Age_diagnosis', 'Years_treatment']

# check for scalar
try:
    scaler  
except NameError:
    # fit new scalar
    if cont_cols:
        scaler = StandardScaler().fit(df[cont_cols].values)
    else:
        scaler = StandardScaler()  # fallback (won't be used)

# categorical one hot encoded column names
if len(categorical_cols) > 0:
    cat_dummy_example = pd.get_dummies(df[categorical_cols], drop_first=True)
    cat_encoded_cols = list(cat_dummy_example.columns)
else:
    cat_encoded_cols = []

# counts
n_binary = len(binary_cols)
n_categorical = len(cat_encoded_cols)
n_cont = len(cont_cols)
n_bin_cat = n_binary + n_categorical  # <-- fixes "n_bin_cat" undefined

# function to produce the processed matrix for PCA/TSNE 
def get_processed_matrix(df_in):
    # binary part: if missing, fill zeros
    bin_part = np.zeros((len(df_in), n_binary))
    for i, col in enumerate(binary_cols):
        if col in df_in.columns:
            bin_part[:, i] = df_in[col].astype(float).values
        else:
            bin_part[:, i] = 0

    # categorical part: try to build same one-hot columns as in cat_encoded_cols
    if n_categorical > 0:
        # if original categorical label columns exist, compute dummies then align
        if all(col in df_in.columns for col in categorical_cols):
            cat_df = pd.get_dummies(df_in[categorical_cols], drop_first=True)
            # ensure all expected encoded cols exist (fill missing with zeros)
            for c in cat_encoded_cols:
                if c not in cat_df.columns:
                    cat_df[c] = 0
            cat_part = cat_df[cat_encoded_cols].values
        else:
            # maybe df_in already contains the one-hot columns (synthetic_df may already be one-hot)
            # select available columns from cat_encoded_cols, fill missing with zeros
            cat_part = np.zeros((len(df_in), n_categorical))
            for j, c in enumerate(cat_encoded_cols):
                if c in df_in.columns:
                    cat_part[:, j] = df_in[c].astype(float).values
                else:
                    cat_part[:, j] = 0
    else:
        cat_part = np.empty((len(df_in), 0))

    # continuous part: scale (use fitted scaler)
    if n_cont > 0:
        # if cont columns present; otherwise make empty array
        if all(col in df_in.columns for col in cont_cols):
            cont_part = scaler.transform(df_in[cont_cols].values)
        else:
            cont_part = np.zeros((len(df_in), n_cont))
    else:
        cont_part = np.empty((len(df_in), 0))

    return np.hstack([bin_part, cat_part, cont_part])

# build processed matrices for real vs synth data
if 'Treat' in df.columns and df['Treat'].isin([0]).any():
    real_controls_df = df[df['Treat'] == 0].reset_index(drop=True)
else:
    real_controls_df = df.copy().reset_index(drop=True)
    print("No real controls found; comparing synthetic controls to whole dataset.")

real_proc = get_processed_matrix(real_controls_df)
synth_proc = get_processed_matrix(synthetic_df)

# sanity shapes
assert real_proc.shape[1] == synth_proc.shape[1], (
    f"Processed real/synth feature dimension mismatch: {real_proc.shape[1]} vs {synth_proc.shape[1]}"
)

# PCA plot
pca = PCA(n_components=2)
combined_for_pca = np.vstack([real_proc, synth_proc])
pca_proj = pca.fit_transform(combined_for_pca)
n_real = real_proc.shape[0]

plt.figure(figsize=(7,6))
plt.scatter(pca_proj[:n_real, 0], pca_proj[:n_real, 1], alpha=0.6, s=10, label="Real controls")
plt.scatter(pca_proj[n_real:, 0], pca_proj[n_real:, 1], alpha=0.6, s=10, label="Synthetic controls")
plt.legend()
plt.title("PCA Comparison: Real vs Synthetic Controls")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_compare.png"), dpi=200)
plt.close()

# T-SNE comp
from sklearn.manifold import TSNE
try:
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    tsne_proj = tsne.fit_transform(combined_for_pca)
    plt.figure(figsize=(7,6))
    plt.scatter(tsne_proj[:n_real,0], tsne_proj[:n_real,1], alpha=0.6, s=10, label="Real controls")
    plt.scatter(tsne_proj[n_real:,0], tsne_proj[n_real:,1], alpha=0.6, s=10, label="Synthetic controls")
    plt.legend()
    plt.title("t-SNE Comparison: Real vs Synthetic Controls")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tsne_compare.png"), dpi=200)
    plt.close()
except Exception as e:
    print("t-SNE failed or skipped:", e)

# per feature comp plots
plot_features = []
plot_types = {}  # map feature -> "binary" | "categorical_label" | "categorical_onehot" | "continuous"

# binary features as listed
for b in binary_cols:
    plot_features.append(b)
    plot_types[b] = "binary"

# categorical: prefer label columns if present, otherwise plot one-hot encoded columns
for c in categorical_cols:
    if c in df.columns:
        plot_features.append(c)
        plot_types[c] = "categorical_label"
    else:
        # fallback: one-hot encoded columns
        for enc in cat_encoded_cols:
            plot_features.append(enc)
            plot_types[enc] = "categorical_onehot"

# continuous features
for c in cont_cols:
    plot_features.append(c)
    plot_types[c] = "continuous"

# subplots
num_features = len(plot_features)
n_cols = min(3, num_features)
n_rows = (num_features + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, feature in enumerate(plot_features):
    ax = axes[i]
    ptype = plot_types[feature]

    if ptype == "binary":
        real_vals = real_controls_df[feature].astype(int).values
        synth_vals = synthetic_df[feature].astype(int).values if feature in synthetic_df.columns else np.zeros(len(synthetic_df))
        real_counts = np.bincount(real_vals, minlength=2) / len(real_vals)
        synth_counts = np.bincount(synth_vals, minlength=2) / len(synth_vals)
        x = np.arange(2)
        ax.bar(x - 0.15, real_counts, width=0.3, label="Real")
        ax.bar(x + 0.15, synth_counts, width=0.3, label="Synthetic")
        ax.set_xticks([0,1])
        ax.set_xlabel(feature)
        ax.set_ylabel("Proportion")
        ax.legend()

    elif ptype == "categorical_label":
        real_counts = real_controls_df[feature].value_counts(normalize=True)
        synth_counts = synthetic_df[feature].value_counts(normalize=True) if feature in synthetic_df.columns else pd.Series(dtype=float)
        categories = sorted(set(real_counts.index).union(synth_counts.index))
        real_freqs = [real_counts.get(cat, 0) for cat in categories]
        synth_freqs = [synth_counts.get(cat, 0) for cat in categories]
        x = np.arange(len(categories))
        ax.bar(x - 0.15, real_freqs, width=0.3, label="Real")
        ax.bar(x + 0.15, synth_freqs, width=0.3, label="Synthetic")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha='right')
        ax.set_xlabel(feature)
        ax.set_ylabel("Proportion")
        ax.legend()

    elif ptype == "categorical_onehot":
        real_vals = real_controls_df[feature].astype(int) if feature in real_controls_df.columns else np.zeros(len(real_controls_df))
        synth_vals = synthetic_df[feature].astype(int) if feature in synthetic_df.columns else np.zeros(len(synthetic_df))
        # treat it like binary proportion
        real_props = np.bincount(real_vals, minlength=2) / len(real_vals)
        synth_props = np.bincount(synth_vals, minlength=2) / len(synth_vals)
        x = np.arange(2)
        ax.bar(x - 0.15, real_props, width=0.3, label="Real")
        ax.bar(x + 0.15, synth_props, width=0.3, label="Synthetic")
        ax.set_xticks([0,1])
        ax.set_xlabel(feature)
        ax.set_ylabel("Proportion (one-hot)")
        ax.legend()

    elif ptype == "continuous":
        real_vals = real_controls_df[feature].values if feature in real_controls_df.columns else np.zeros(len(real_controls_df))
        synth_vals = synthetic_df[feature].values if feature in synthetic_df.columns else np.zeros(len(synthetic_df))
        ax.hist(real_vals, bins=30, alpha=0.6, label="Real")
        ax.hist(synth_vals, bins=30, alpha=0.6, label="Synthetic")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        ax.legend()

    ax.set_title(f"{feature}: Real vs Synthetic")

# hide any unused axes
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_comparisons.png"), dpi=200)
plt.close()

print("Visualization plots saved to:", OUT_DIR)


# graph separately : 

# per feature plots

os.makedirs(OUT_DIR, exist_ok=True)

TOP_N_CATS = 10  # show top 10 categories max for readability

for feature in plot_features:
    ptype = plot_types[feature]

    real_vals = real_controls_df[feature].values if feature in real_controls_df.columns else np.zeros(len(real_controls_df))
    synth_vals = synthetic_df[feature].values if feature in synthetic_df.columns else np.zeros(len(synthetic_df))

    # cat dist cleaner function
    def plot_categorical(data, title, color, filename):
        vc = pd.Series(data).value_counts(normalize=True)
        if len(vc) > TOP_N_CATS:
            top_cats = vc.nlargest(TOP_N_CATS)
            other_sum = 1 - top_cats.sum()
            top_cats.loc["Other"] = other_sum
            vc = top_cats
        plt.figure(figsize=(6, 4))
        vc.plot(kind="bar", color=color)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Proportion")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()

    # plot real data
    if ptype in ["binary", "categorical_onehot", "categorical_label"]:
        if ptype in ["binary", "categorical_onehot"]:
            vals = real_vals.astype(int)
            n_categories = int(vals.max()) + 1 if len(vals) > 0 else 2
            counts = np.bincount(vals, minlength=n_categories) / len(vals)
            plt.figure(figsize=(6, 4))
            plt.bar(np.arange(n_categories), counts, color="steelblue", width=0.6)
            plt.xticks(np.arange(n_categories))
            plt.ylabel("Proportion")
            plt.xlabel(feature)
            plt.title(f"{feature} — Real Data")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"{feature}_Real.png"), dpi=200)
            plt.close()
        else:
            plot_categorical(real_vals, f"{feature} — Real Data", "steelblue",
                             os.path.join(OUT_DIR, f"{feature}_Real.png"))

    elif ptype == "continuous":
        plt.figure(figsize=(6, 4))
        plt.hist(real_vals, bins=30, color="steelblue", alpha=0.8)
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.title(f"{feature} — Real Data")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{feature}_Real.png"), dpi=200)
        plt.close()

    # plot synth data
    if ptype in ["binary", "categorical_onehot", "categorical_label"]:
        if ptype in ["binary", "categorical_onehot"]:
            vals = synth_vals.astype(int)
            n_categories = int(vals.max()) + 1 if len(vals) > 0 else 2
            counts = np.bincount(vals, minlength=n_categories) / len(vals)
            plt.figure(figsize=(6, 4))
            plt.bar(np.arange(n_categories), counts, color="coral", width=0.6)
            plt.xticks(np.arange(n_categories))
            plt.ylabel("Proportion")
            plt.xlabel(feature)
            plt.title(f"{feature} — Synthetic Data")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"{feature}_Synthetic.png"), dpi=200)
            plt.close()
        else:
            plot_categorical(synth_vals, f"{feature} — Synthetic Data", "coral",
                             os.path.join(OUT_DIR, f"{feature}_Synthetic.png"))

    elif ptype == "continuous":
        plt.figure(figsize=(6, 4))
        plt.hist(synth_vals, bins=30, color="coral", alpha=0.8)
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.title(f"{feature} — Synthetic Data")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{feature}_Synthetic.png"), dpi=200)
        plt.close()

print(f"✅ Separate feature plots saved to: {OUT_DIR}")

# additional synth data checks


# correlation heatmaps 
if len(cont_idx) > 0:
    cont_cols = [feature_names[i] for i in cont_idx]

    # select continuous features by column name
    real_cont = real_compare[cont_cols]
    synth_cont = synthetic_df[cont_cols]
    plt.figure(figsize=(8,4))
    sns.heatmap(real_cont.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap - Real Data (Continuous)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "corr_real.png"), dpi=200)
    plt.close()
    
    plt.figure(figsize=(8,4))
    sns.heatmap(synth_cont.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap - Synthetic Data (Continuous)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "corr_synthetic.png"), dpi=200)
    plt.close()

# downstream model check: train on synthetic, test on real
X_train = synthetic_df[feature_names].values
y_train = synthetic_df['Treat'].values
X_test = df[feature_names].values
y_test = df['Treat'].values

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest trained on synthetic, tested on real: Accuracy={acc:.4f}")

# diversity check: unique rows in synthetic vs real
unique_synth = len(np.unique(synthetic_df[feature_names].values, axis=0))
unique_real = len(np.unique(real_compare, axis=0))
print(f"Synthetic unique rows: {unique_synth}, Real unique rows: {unique_real}")

print("Additional synthetic data checks completed. Plots saved to:", OUT_DIR)





