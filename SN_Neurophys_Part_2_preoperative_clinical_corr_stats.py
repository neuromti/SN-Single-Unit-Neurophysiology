# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:16:55 2026

@author: tameem
"""

#%% import and prepare main df with clinical scores

from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

def load_updrs3(updrs_3_on_path):
    #load updrs3
    df_updrs3= pd.read_csv(updrs_3_on_path, index_col="PatientID").sort_values(by="PatientID")
    print("Missing values before processing:\n", df_updrs3.isna().sum())
    
    df_updrs3.drop(columns=["MDS-UPDRS_Dyskinesien", 
                                    "MDS-UPDRS_Einfluss",
                                    "MDS-H&Y"],inplace = True)
    
    print("Total missing values before fillna:", df_updrs3.isna().sum().sum())
    df_updrs3= df_updrs3.fillna(df_updrs3.median())
    print("Total missing values after fillna:", df_updrs3.isna().sum().sum())
    df_updrs3 = df_updrs3.reset_index()
    
    # --------------------------------- TREMOR ----------------------------------

    
    df_updrs3["right_tremor"] = df_updrs3['MDS-UPDRS_3.15R'] +\
        df_updrs3['MDS-UPDRS_3.16R'] +\
        df_updrs3['MDS-UPDRS_3.17ROE'] +\
        df_updrs3['MDS-UPDRS_3.17RUE']
        
        
    df_updrs3["left_tremor"] = df_updrs3['MDS-UPDRS_3.15L']  +\
        df_updrs3['MDS-UPDRS_3.16L'] +\
        df_updrs3['MDS-UPDRS_3.17LOE'] +\
        df_updrs3['MDS-UPDRS_3.17LUE'] 
        
        
    #df_updrs3["tremor"] = df_updrs3["right_tremor"] + df_updrs3["left_tremor"]
    
    # --------------------------------- BRADIKYNESIA ----------------------------

    
    df_updrs3["right_bradikynesia"] =  df_updrs3["MDS-UPDRS_3.4R"] +\
        df_updrs3["MDS-UPDRS_3.5R"] +\
        df_updrs3["MDS-UPDRS_3.6R"] +\
        df_updrs3["MDS-UPDRS_3.7R"] +\
        df_updrs3["MDS-UPDRS_3.8R"] + df_updrs3["MDS-UPDRS_3.14"]
    
    
    df_updrs3["left_bradikynesia"] =  df_updrs3["MDS-UPDRS_3.4L"] +\
        df_updrs3["MDS-UPDRS_3.5L"] +\
        df_updrs3["MDS-UPDRS_3.6L"] +\
        df_updrs3["MDS-UPDRS_3.7L"] +\
        df_updrs3["MDS-UPDRS_3.8L"] + df_updrs3["MDS-UPDRS_3.14"]
    
            
    #df_updrs3["bradikynesia"] = df_updrs3["right_bradikynesia"] + df_updrs3["left_bradikynesia"] -  df_updrs3["MDS-UPDRS_3.14"]
    
    # --------------------------------- RIGIDITY --------------------------------

    
    df_updrs3["right_rigidity"] = df_updrs3["MDS-UPDRS_3.3ROE"] +\
        df_updrs3["MDS-UPDRS_3.3RUE"]
    
    
    df_updrs3["left_rigidity"] = df_updrs3["MDS-UPDRS_3.3LOE"] +\
        df_updrs3["MDS-UPDRS_3.3LUE"]
    
    
    # --------------------------------- AXIAL -----------------------------------

    
    df_updrs3["right_axial"] = df_updrs3["MDS-UPDRS_3.1"] +\
        df_updrs3["MDS-UPDRS_3.2"] +\
        df_updrs3["MDS-UPDRS_3.8R"] +\
        df_updrs3["MDS-UPDRS_3.9"] +\
        df_updrs3["MDS-UPDRS_3.10"] +\
        df_updrs3["MDS-UPDRS_3.11"] +\
        df_updrs3["MDS-UPDRS_3.12"] +\
        df_updrs3["MDS-UPDRS_3.13"]
                    
                    
    df_updrs3["left_axial"] = df_updrs3["MDS-UPDRS_3.1"] +\
        df_updrs3["MDS-UPDRS_3.2"]  +\
        df_updrs3["MDS-UPDRS_3.8L"] +\
        df_updrs3["MDS-UPDRS_3.9"] +\
        df_updrs3["MDS-UPDRS_3.10"] +\
        df_updrs3["MDS-UPDRS_3.11"] +\
        df_updrs3["MDS-UPDRS_3.12"] +\
        df_updrs3["MDS-UPDRS_3.13"]
                    
    # ---------------------------------------------------------------------------

    
    df_updrs3["all_right"] = df_updrs3["right_rigidity"] +\
        df_updrs3["right_bradikynesia"]  +\
        df_updrs3["right_axial"] +\
        df_updrs3["right_tremor"]                
    
    
    df_updrs3["all_left"] = df_updrs3["left_rigidity"] +\
        df_updrs3["left_bradikynesia"]  +\
        df_updrs3["left_axial"] +\
        df_updrs3["left_tremor"]
                
                
    # ---------------------------------------------------------------------------



    df_updrs3 = df_updrs3.rename(columns={"PatientID": "patient_id"})
    df_temp  = pd.DataFrame()
    df_temp["patient_id"] = pd.concat([df_updrs3["patient_id"],df_updrs3["patient_id"]])
    df_temp["tremor_score"] = pd.concat([df_updrs3["right_tremor"],df_updrs3["left_tremor"]])
    df_temp["bradikynesia_score"] = pd.concat([df_updrs3["right_bradikynesia"],df_updrs3["left_bradikynesia"]])
    df_temp["rigidity_score"] = pd.concat([df_updrs3["right_rigidity"],df_updrs3["left_rigidity"]])
    df_temp["axial_score"] = pd.concat([df_updrs3["right_axial"],df_updrs3["left_axial"]])
    df_temp["all_score"] = pd.concat([df_updrs3["all_right"],df_updrs3["all_left"]])
    df_temp["hem"] = "R" 
    df_temp = df_temp.reset_index(drop=True)  
    df_temp.loc[0:df_updrs3.shape[0]-1, "hem"] = "L"  
    
    return df_temp


if __name__ == '__main__':
    
    updrs_3_on_path = ("D:\\new code\\nigraphysCLINscores.csv")
    df_updrs3= pd.read_csv(updrs_3_on_path, index_col="PatientID").sort_values(by="PatientID")
    
    df_cellclinics = load_updrs3(updrs_3_on_path)
    print("Missing values in final df_cellclinics:", df_cellclinics.isna().sum())


#%% Load cell data and merge with clinical scores

clustered_cells = pd.read_csv('D:\\new code\\clustered_cells_pub.csv')
clustered_cells['hemisphere'] = clustered_cells['hemisphere'].map({'right': 'R', 'left': 'L'})
df_cellclinics['hem'] = df_cellclinics['hem'].map({'R': 'R', 'L': 'L'})

# Merge DataFrames on 'patient_id' and 'hemisphere'
merged_cellclinics_df = pd.merge(clustered_cells, df_cellclinics, left_on=['patient_id', 'hemisphere'], right_on=['patient_id', 'hem'], how='inner')

# Count cells per cluster
cell_counts = merged_cellclinics_df.groupby(['patient_id', 'hemisphere', 'cluster_name']).size().reset_index(name='cell_count')

# Pivot to create a column for each cluster's cell count
cell_count_pivot = cell_counts.pivot_table(index=['patient_id', 'hemisphere'], columns='cluster_name', values='cell_count', fill_value=0).reset_index()

# Flatten the columns after pivoting
cell_count_pivot.columns = [f'cell_count_cluster_{col}' if isinstance(col, str) else col for col in cell_count_pivot.columns]

# Print the cell count DataFrame to verify
print(cell_count_pivot.head())

cluster_means = merged_cellclinics_df.groupby(['patient_id', 'hemisphere', 'cluster_name']).agg({
    'firing': 'mean',
    'mean_isi': 'mean',
    'burst_index': 'mean',
    'fano_factor': 'mean', 
    'lv': 'mean',
    'renyi_entropy': 'mean'
}).reset_index()

# Pivot this data to get each cluster's mean firing rate as a separate column
cluster_pivot = cluster_means.pivot_table(
    index=['patient_id', 'hemisphere'],
    columns='cluster_name',
    values=['firing', 'mean_isi', 'burst_index', 'fano_factor', 'lv', 'renyi_entropy'],
    aggfunc='first'
).reset_index()

# Flatten the MultiIndex in columns after pivoting
new_columns = []
for col_tuple in cluster_pivot.columns:
    if col_tuple[0] in ['patient_id', 'hemisphere']:
        new_columns.append(col_tuple[0])
    else:
        new_columns.append(f"{col_tuple[0]}_{col_tuple[1]}")  # Combines type (firing, mean_isi) with cluster_name

cluster_pivot.columns = new_columns

# Assuming clinical scores are still part of `merged_cellclinics_df` and you want all scores along with the firing rates and mean ISI
clinical_scores = merged_cellclinics_df[['patient_id', 'hemisphere', 'tremor_score', 'bradikynesia_score', 'rigidity_score', 'axial_score', 'all_score']].drop_duplicates()

merged_cellclinics_df.to_csv('merged_cellclinics_df.csv', index=False)

final_data = pd.merge(clinical_scores, cluster_pivot, on=['patient_id', 'hemisphere'], how='inner')

# Correcting column names after resetting the index
cell_count_pivot.reset_index(inplace=True)

# Renaming columns to the expected names
cell_count_pivot.rename(columns={
    'cell_count_cluster_patient_id': 'patient_id',
    'cell_count_cluster_hemisphere': 'hemisphere'
}, inplace=True)

# Check if the columns are named correctly now
print("Columns after renaming:", cell_count_pivot.columns)

# Now, attempt the merge again
try:
    final_cellclinics_df = pd.merge(final_data, cell_count_pivot, on=['patient_id', 'hemisphere'], how='inner')
    print("Merge successful, here's the head of the final merged DataFrame:")
    print(final_cellclinics_df.head())
except Exception as e:
    print("Failed to merge:", e)

# Save the final DataFrame to a file if the merge was successful
final_cellclinics_df.to_csv('final_cellclinics_dfnew.csv', index=False)

unique_patient_count = final_cellclinics_df['patient_id'].nunique()
print(unique_patient_count)

# Replace spaces in column names with underscores for compatibility with formulas
final_cellclinics_df.columns = final_cellclinics_df.columns.str.replace(" ", "_")


#%% Imputing (each row separate) with Bayesian modeling - ALL FEATURES

import pandas as pd
import numpy as np
import pymc as pm
import statsmodels.formula.api as smf
import arviz as az

# Step 1: Check for missing data in the original DataFrame
print("Missing data summary before Bayesian imputation:")
print(final_cellclinics_df.isnull().sum())

# Step 2: Create a copy of the original DataFrame to avoid modification
final_cellclinics_df_copy = final_cellclinics_df.copy()

# Step 3: Define a function for Bayesian imputation using PyMC
def bayesian_imputation(df, features):
    """
    Imputes missing data for specified features using Bayesian models.

    Parameters:
    df: pd.DataFrame - Input DataFrame with missing data
    features: list - List of feature names to impute (e.g., 'firing', 'burst_index')

    Returns:
    df: pd.DataFrame - DataFrame with imputed values
    traces: dict - Dictionary of traces for each feature and cluster
    """
    traces = {}  # Store traces for inspection

    for feature in features:
        for cluster in range(1, 5):  # Assuming 4 clusters
            col_name = f"{feature}_Cluster_{cluster}"
            observed_data = df[col_name].dropna().values
            missing_mask = df[col_name].isnull()
            num_missing = int(missing_mask.sum())

            if num_missing > 0:
                print(f"Imputing missing values for {col_name}...")

                # Bayesian model
                with pm.Model() as model:
                    # Priors for mean and standard deviation
                    mu = pm.Normal('mu', mu=0, sigma=10)
                    sigma = pm.HalfNormal('sigma', sigma=5)

                    # Observed data likelihood
                    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=observed_data)

                    # Impute missing values
                    y_mis = pm.Normal('y_mis', mu=mu, sigma=sigma, shape=num_missing)

                    # Sampling (ensure return_inferencedata=True)
                    trace = pm.sample(draws=1000, chains=4, cores=1, random_seed=50,
                                      return_inferencedata=True, progressbar=True)

                # Store the trace for inspection
                traces[f"{col_name}"] = trace

                # Extract posterior mean for imputed values
                imputed_values = trace.posterior['y_mis'].mean(axis=(0, 1)).values

                # Fill missing values in the DataFrame
                df.loc[missing_mask, col_name] = imputed_values

    return df, traces


# Step 4: Apply Bayesian imputation to the selected features
features = ['firing', 'burst_index', 'fano_factor', 'lv', 'renyi_entropy', 'mean_isi']
final_cellclinics_df_imputed, imputation_traces = bayesian_imputation(final_cellclinics_df_copy, features)

# Step 5: Check if there are still missing values after imputation
print("\nMissing data after Bayesian imputation:")
print(final_cellclinics_df_imputed.isnull().sum())


#%% Standard OLS multiple regression on Bayesian imputed data

import statsmodels.formula.api as smf

def run_multiple_regression(df, feature_columns, subscore):
    """
    Runs a standard multiple regression model.

    Parameters:
    df: pd.DataFrame - Input DataFrame
    feature_columns: list - Columns representing features
    subscore: str - Dependent variable (clinical subscore)

    Returns:
    model_fit: statsmodels object - Fitted regression model
    """
    # Build the formula for standard regression (without random effects)
    formula = f"{subscore} ~ " + " + ".join(feature_columns)
    model = smf.ols(formula, data=df)
    model_fit = model.fit()
    return model_fit

# Step 7: List of clinical scores to use as predictors
clinical_scores = ['tremor_score', 'bradikynesia_score', 'rigidity_score', 'axial_score', 'all_score']
features = ['firing', 'burst_index', 'fano_factor', 'lv', 'renyi_entropy', 'mean_isi']

ols_results = {}

for feature in features:
    print(f"\nRunning models for {feature} with Bayesian-imputed data...")

    feature_columns = [f"{feature}_Cluster_{i}" for i in range(1, 5)]  # Cluster columns

    for subscore in clinical_scores:
        print(f"Model for {feature} and {subscore}")
        model_name = f"{feature}_model_for_{subscore}"
        model = run_multiple_regression(final_cellclinics_df_imputed, feature_columns, subscore)
        ols_results[model_name] = model

        # Save summaries
        with open(f'{model_name}_summary.txt', 'w') as f:
            f.write(model.summary().as_text())
            
# Step 9: Print model summaries for each feature and each clinical subscore
for model_name, model in ols_results.items(): 
    print(f"\nModel results for {model_name} with Bayesian-imputed data:\n")
    print(model.summary())


#%% OLS regression output visualization 

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.multitest import multipletests
import pandas as pd

# Extract p-values for each feature, cluster, and subscore (OLS results)
p_values_by_cluster_OLS = {}

for model_name, model in ols_results.items():
    feature, subscore = model_name.split("_model_for_")
    clusters = [f"{feature}_Cluster_{i}" for i in range(1, 5)]  # Assuming 4 clusters

    p_values_by_cluster_OLS[model_name] = {}

    for cluster in clusters:
        try:
            # Extract p-value for the specific cluster coefficient
            p_value = model.pvalues[cluster]
            p_values_by_cluster_OLS[model_name][cluster] = p_value
        except KeyError:
            # If cluster not in model coefficients, assign NaN
            p_values_by_cluster_OLS[model_name][cluster] = np.nan

# Apply FDR correction (Benjamini-Hochberg) to p-values
adjusted_p_values_OLS = {}
for model_name, cluster_p_values in p_values_by_cluster_OLS.items():
    raw_p_values = list(cluster_p_values.values())
    _, corrected_p, _, _ = multipletests(raw_p_values, method="fdr_bh")
    adjusted_p_values_OLS[model_name] = dict(zip(cluster_p_values.keys(), corrected_p))


# Visualization function (same format as earlier code)
def visualize_coefficients_and_pvalues_with_correction(
    model_results, features, clinical_scores,
    raw_pvalues, corrected_pvalues, significance_level=0.05):
    """
    Visualize coefficients and p-values (raw and corrected) of feature contributions to each clinical subscore.
    """

    for subscore in clinical_scores:
        feature_coeffs = {}
        feature_raw_pvalues = {}
        feature_corrected_pvalues = {}

        for feature in features:
            model_name = f"{feature}_model_for_{subscore}"
            if model_name in model_results:
                model = model_results[model_name]
                # Extract coefficients (1:5 for the four clusters)
                coeffs = model.params[1:5].values
                feature_coeffs[feature] = coeffs

                # Extract raw and corrected p-values
                raw_pvalues_for_feature = raw_pvalues[model_name]
                corrected_pvalues_for_feature = corrected_pvalues[model_name]

                feature_raw_pvalues[feature] = list(raw_pvalues_for_feature.values())
                feature_corrected_pvalues[feature] = list(corrected_pvalues_for_feature.values())

        # Flatten data for plotting
        flattened_coeffs = []
        flattened_raw_pvalues = []
        flattened_corrected_pvalues = []
        flattened_labels = []
        bar_colors_coeffs = []  # Colors for coefficients
        bar_colors_raw_pvalues = []  # Colors for raw p-values

        for feature in features:
            for i in range(4):  # Assuming 4 clusters per feature
                label = f"{feature}_Cluster_{i+1}"
                flattened_labels.append(label)
                coeff = feature_coeffs[feature][i]
                raw_pval = feature_raw_pvalues[feature][i]

                flattened_coeffs.append(coeff)
                flattened_raw_pvalues.append(raw_pval)
                flattened_corrected_pvalues.append(feature_corrected_pvalues[feature][i])

                # Set colors for coefficients and p-values
                bar_colors_coeffs.append("tomato" if raw_pval < significance_level else "dimgrey")
                bar_colors_raw_pvalues.append("tomato" if raw_pval < significance_level else "darkgrey")

        # Create figure with three subplots
        fig, axs = plt.subplots(3, 1, figsize=(16, 15), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

        # Coefficients Plot
        axs[0].bar(flattened_labels, flattened_coeffs, color=bar_colors_coeffs)
        axs[0].axhline(y=0, color="black", linestyle="--")
        axs[0].set_title(f"Coefficients for {subscore.replace('_', ' ')}", fontsize=20, weight='bold')
        axs[0].set_ylabel("Coefficient", fontsize=12, weight='bold')
        axs[0].tick_params(axis='x', rotation=45)
        axs[0].tick_params(axis='y', labelsize=10)
        for label in axs[0].yaxis.get_ticklabels():
            label.set_fontweight('bold')

        # Raw P-values Plot
        axs[1].bar(flattened_labels, flattened_raw_pvalues, color=bar_colors_raw_pvalues)
        axs[1].axhline(y=significance_level, color="black", linestyle="--", linewidth=2, label="Significance Threshold (0.05)")
        axs[1].axhline(y=0.1, color="black", linestyle=":", linewidth=2, label="Threshold (0.1)")  # New line
        axs[1].set_title(f"Raw P-values for {subscore.replace('_', ' ')}", fontsize=20, weight='bold')
        axs[1].set_ylabel("P-value", fontsize=12, weight='bold')
        axs[1].set_ylim(0, 1.05)
        axs[1].tick_params(axis='x', rotation=90)
        axs[1].tick_params(axis='y', labelsize=10)
        axs[1].legend(fontsize=10, prop={'weight': 'bold'})

        # Corrected P-values Plot
        axs[2].bar(flattened_labels, flattened_corrected_pvalues, color="lightgrey")
        axs[2].axhline(y=significance_level, color="black", linestyle="--", linewidth=2, label="Corrected Threshold (0.05)")
        axs[2].axhline(y=0.1, color="black", linestyle=":", linewidth=2, label="Corrected Threshold (0.1)")
        axs[2].set_title(f"Corrected P-values for {subscore.replace('_', ' ')}", fontsize=20, weight='bold')
        axs[2].set_ylabel("Corrected P-value", fontsize=12, weight='bold')
        axs[2].set_ylim(0, 1.05)
        axs[2].tick_params(axis='x', labelsize=10, rotation=90)
        axs[2].tick_params(axis='y', labelsize=10)
        axs[2].legend(fontsize=10, prop={'weight': 'bold'})

        # Highlight significant corrected p-values
        for bar, pval in zip(axs[2].containers[0], flattened_corrected_pvalues):
            if pval < significance_level:
                bar.set_color("tomato")

        plt.tight_layout()
        plt.show()


# Call the function with the OLS results
visualize_coefficients_and_pvalues_with_correction(
    model_results=ols_results,
    features=features,
    clinical_scores=clinical_scores,
    raw_pvalues=p_values_by_cluster_OLS,
    corrected_pvalues=adjusted_p_values_OLS
)


#%% OLS regression output visualization: forest plots with standardization (β*) - Supp Figure 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
import matplotlib.transforms as mtransforms

def _beta_star_and_se(model, predictors):
    """
    Compute standardized coefficients β* and their SE for the given predictor names.
    β*_j = β_j * SD(X_j)/SD(Y)
    SE(β*_j) = SE(β_j) * SD(X_j)/SD(Y)
    Returns arrays aligned to `predictors`. Missing predictors are skipped.
    """
    exog_names = model.model.exog_names
    name_to_col = {n: i for i, n in enumerate(exog_names)}

    keep = [p for p in predictors if p in name_to_col]
    if not keep:
        return np.array([]), np.array([]), [], int(model.df_resid)

    idxs = [name_to_col[p] for p in keep]
    X = model.model.exog
    y = model.model.endog

    sd_y = np.std(y, ddof=1)
    if not np.isfinite(sd_y) or sd_y == 0:
        sd_y = 1.0  # fall back to avoid blow-ups

    sd_x = np.std(X[:, idxs], axis=0, ddof=1)
    scale = np.where(np.isfinite(sd_x), sd_x, 0.0) / sd_y

    beta_raw = model.params[idxs].values
    se_raw   = model.bse[idxs].values

    beta_star = beta_raw * scale
    se_star   = se_raw   * scale
    return beta_star, se_star, keep, int(model.df_resid)

def visualize_forest_plots_standardized_coefficients(
    model_results, features, clinical_scores,
    raw_pvalues, corrected_pvalues, significance_level=0.05,
    whisker="ci",              # "ci" for 95% CI (default), "se" for ±1 SE
    right_gutter=0.32          # fraction of figure width reserved for annotations
):
    """
    Forest plots of standardized coefficients (β*) with either 95% CI or ±1 SE whiskers.
    - Reserves a consistent right-side gutter for annotations (no overlap).
    - Annotations are placed in axes-fraction (stable) coordinates.
    - Top/right spines hidden for a clean, non-boxed look.
    """
    for subscore in clinical_scores:
        labels, means, errs, rawps, corrps, colors = [], [], [], [], [], []

        for feature in features:
            model_key = f"{feature}_model_for_{subscore}"
            if model_key not in model_results:
                continue
            model = model_results[model_key]
            predictors = [f"{feature}_Cluster_{i}" for i in range(1, 5)]

            beta_star, se_star, kept_preds, df = _beta_star_and_se(model, predictors)
            if beta_star.size == 0:
                continue

            raw_p_dict  = raw_pvalues.get(model_key, {})
            corr_p_dict = corrected_pvalues.get(model_key, {})

            if whisker == "ci":
                tcrit = student_t.ppf(0.975, df)
                half_width = se_star * tcrit
            else:
                half_width = se_star

            labels.extend([p.replace('_', ' ') for p in kept_preds])
            means.extend(beta_star.tolist())
            errs.extend(half_width.tolist())
            rawps.extend([float(raw_p_dict.get(p, np.nan))  for p in kept_preds])
            corrps.extend([float(corr_p_dict.get(p, np.nan)) for p in kept_preds])
            colors.extend(["tomato" if float(raw_p_dict.get(p, np.inf)) < significance_level else "dimgrey"
                           for p in kept_preds])

        if not labels:
            continue

        # Reverse for top-down plotting
        labels = labels[::-1]
        means  = means[::-1]
        errs   = errs[::-1]
        rawps  = rawps[::-1]
        corrps = corrps[::-1]
        colors = colors[::-1]

        # Figure with extra right margin reserved for annotations
        fig, ax = plt.subplots(figsize=(12, 0.5*len(labels) + 2), constrained_layout=False)
        # Reserve a right gutter (same layout across panels)
        fig.subplots_adjust(right=1.0 - right_gutter)

        # Clean spines (avoid the “box” look)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        y = np.arange(len(labels))

        # Draw points + whiskers
        for i in range(len(labels)):
            ax.errorbar(means[i], y[i], xerr=errs[i], fmt='o',
                        color='black', ecolor=colors[i], elinewidth=3, capsize=5)

        # Axes limits with a little symmetric padding around the furthest whisker
        x_absmax = np.nanmax(np.abs(np.array(means) + np.array(errs)))
        pad = 0.12 * (x_absmax if np.isfinite(x_absmax) and x_absmax > 0 else 1.0)
        ax.set_xlim(-x_absmax - pad, x_absmax + pad)

        ax.axvline(x=0, color='black', linestyle='--')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10, fontweight='bold')

        if whisker == "ci":
            ax.set_xlabel("Standardized coefficient (β*) with 95% CI", fontsize=14, fontweight='bold')
        else:
            ax.set_xlabel("Standardized coefficient (β*) ± 1 SE", fontsize=12, fontweight='bold')

        ax.set_title(f"Forest Plot for {subscore.replace('_', ' ')}", fontsize=16, fontweight='bold')

        # ---- Right-column annotations (stable, non-overlapping) ----
        # Place text in a fixed right margin using blended transform:
        # x in axes-fraction (e.g., 1.02 = just outside axes), y in data coords (row index)
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

        for i in range(len(labels)):
            p_raw  = rawps[i]
            p_corr = corrps[i]

            # Build the annotation string: dagger for trend, then p-values if p≤.10
            anno_parts = []
            if np.isfinite(p_raw) and (0.05 < p_raw <= 0.10):
                anno_parts.append("†")
            if np.isfinite(p_raw) and p_raw <= 0.10:
                if np.isfinite(p_corr):
                    anno_parts.append(f"p={p_raw:.3f}, p_corr={p_corr:.3f}")
                else:
                    anno_parts.append(f"p={p_raw:.3f}")

            if anno_parts:
                ax.text(1.02, y[i], " ".join(anno_parts),
                        transform=trans, ha='left', va='center', fontsize=9,
                        fontweight='bold', color='black', clip_on=False)

        
        plt.show()
        

visualize_forest_plots_standardized_coefficients(
     model_results=ols_results,
     features=features,
     clinical_scores=clinical_scores,
     raw_pvalues=p_values_by_cluster_OLS,
     corrected_pvalues=adjusted_p_values_OLS,
     significance_level=0.05,
     whisker="ci"   # or "se"
 )



#%% Heatmaps of standardized coefficients (β*) - Figure 8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def _beta_star_from_model(model, df, y_col, x_cols):
    """
    β*_j = β_j * SD(X_j) / SD(Y), computed from the provided data frame.
    Drops Intercept/const automatically. Returns a pd.Series indexed by x_cols.
    """
    betas = model.params.drop(labels=[c for c in model.params.index
                                      if c.lower() in ("intercept", "const")],
                              errors='ignore')
    betas = betas.reindex(x_cols)  # ensure column order
    sd_y = df[y_col].std(ddof=1)
    sd_x = df[x_cols].std(ddof=1).replace(0, np.nan)

    if not np.isfinite(sd_y) or sd_y == 0:
        return pd.Series([np.nan]*len(x_cols), index=x_cols)

    return betas * (sd_x / sd_y)

def visualize_beta_star_heatmaps(
    model_results,          # dict: model_name -> fitted statsmodels OLS
    features,               # e.g. ['firing','burst_index','fano_factor','lv','renyi_entropy','mean_isi']
    clinical_scores,        # e.g. ['tremor_score','bradikynesia_score','rigidity_score','axial_score','all_score']
    raw_pvalues,            # dict: model_name -> dict({f'feature_Cluster_i': p, ...})
    corrected_pvalues,      # same keys as raw_pvalues
    dataset,                # DataFrame used to fit the models
    cluster_count=4,
    cmap="coolwarm",
    decimals_beta=2,
    decimals_p=3,
    alpha=0.05,
    trend=0.10
):
    # pretty names (optional)
    pretty = {f: ('local variability' if f=='lv' else
                  'firing rate' if f=='firing' else
                  f.replace('_',' ')) for f in features}

    for subscore in clinical_scores:
        rows, annots, row_labels = [], [], []

        for f in features:
            model_name = f"{f}_model_for_{subscore}"
            if model_name not in model_results:
                continue

            x_cols = [f"{f}_Cluster_{i}" for i in range(1, cluster_count+1)]
            beta_star = _beta_star_from_model(model_results[model_name], dataset, subscore, x_cols)

            # p-values in the same order (fall back to NaN if missing)
            rp_dict = raw_pvalues.get(model_name, {})
            cp_dict = corrected_pvalues.get(model_name, {})
            raw_list  = [float(rp_dict.get(c, np.nan)) for c in x_cols]
            corr_list = [float(cp_dict.get(c, np.nan)) for c in x_cols]

            rows.append(beta_star.values)
            row_labels.append(pretty[f])

            # annotation strings for EVERY cell: β*, p, pFDR
            this_row_ann = []
            for j in range(cluster_count):
                b = beta_star.values[j]
                pr = raw_list[j]
                pc = corr_list[j]

                b_txt  = f"β*={b:.{decimals_beta}f}" if np.isfinite(b) else "β*=NA"
                p_txt  = f"p={pr:.{decimals_p}f}"    if np.isfinite(pr) else "p=NA"
                pf_txt = f"pFDR={pc:.{decimals_p}f}" if np.isfinite(pc) else "pFDR=NA"

                this_row_ann.append(f"{b_txt}\n{p_txt}\n{pf_txt}")
            annots.append(this_row_ann)

        if not rows:
            continue

        beta_mat = np.vstack(rows)
        df_heat = pd.DataFrame(beta_mat, index=row_labels,
                               columns=[f"Cluster {i}" for i in range(1, cluster_count+1)])
        annot_df = pd.DataFrame(annots, index=row_labels, columns=df_heat.columns)

        plt.figure(figsize=(9, max(8.5, len(row_labels))))
        ax = sns.heatmap(df_heat, annot=False, cmap=cmap, center=0.0,
                         cbar_kws={'label': 'Standardized coefficient (β*)'},
                         linewidths=0.5)

        # titles & ticks
        ax.set_title(f"β* heatmap: {subscore.replace('_',' ')}", fontsize=16, weight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='center', fontsize=14, weight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14, weight='bold')

        # bold colorbar ticks/label
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(14)
        cbar.ax.yaxis.label.set_weight("bold")
        cbar.ax.tick_params(labelsize=12)
        for lab in cbar.ax.get_yticklabels():
            lab.set_weight("bold")

        # draw annotations; emphasize by raw p (bold ≤ α, semibold for trends)
        for i in range(df_heat.shape[0]):
            for j in range(df_heat.shape[1]):
                txt = annot_df.iloc[i, j]
                # parse raw p for styling
                try:
                    # assumes the second line is like "p=0.012"
                    pr_line = txt.splitlines()[1]
                    pr_val = float(pr_line.split('=')[1])
                except Exception:
                    pr_val = np.nan

                if np.isfinite(pr_val) and pr_val <= alpha:
                    weight = "semibold"
                elif np.isfinite(pr_val) and pr_val <= trend:
                    weight = "semibold"
                else:
                    weight = "normal"

                ax.text(j + 0.5, i + 0.5, txt,
                        ha="center", va="center",
                        fontsize=12, fontweight=weight, color='black')

        plt.tight_layout()
        plt.show()
        

visualize_beta_star_heatmaps(
    model_results=ols_results,
    features=features,
    clinical_scores=clinical_scores,
    raw_pvalues=p_values_by_cluster_OLS,
    corrected_pvalues=adjusted_p_values_OLS,
    dataset=final_cellclinics_df_imputed,   # the dataframe you fit the models on
    cluster_count=4,                        # change if you used a different number
    cmap="coolwarm",                        # optional
    decimals_beta=2,                        # optional formatting
    decimals_p=3,                           # optional formatting
    alpha=0.05,                             # bold if p ≤ alpha
    trend=0.10                              # semibold if alpha < p ≤ trend
)


#%% Complementary Spearman corr plots (with yep transform) (OLS results) - Figure 8

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn.preprocessing import PowerTransformer


def visualize_significant_correlations_OLS_yeo_with_ranks(
    features, clinical_scores, raw_pvalues, dataset, significance_level=0.05,
    x_jitter_target=0.01, y_jitter_target=0.01):
    """
    Visualize raw, ranked, and Yeo-Johnson transformed data for significant correlations (OLS).
    Adds jitter selectively to specified plots for visualization only (not for stats).
    """
    significant_correlations = []

    # Identify significant correlations based on raw p-values
    for feature in features:
        for subscore in clinical_scores:
            model_name = f"{feature}_model_for_{subscore}"
            if model_name in raw_pvalues:
                for cluster, pval in raw_pvalues[model_name].items():  # Use raw p-values
                    if pval < significance_level:
                        significant_correlations.append((cluster, subscore))

    print(f"Significant correlations (OLS, Raw P-values): {significant_correlations}")

    # Generate subplots for each significant correlation
    for cluster, subscore in significant_correlations:
        if cluster in dataset.columns and subscore in dataset.columns:
            # Extract raw data (Original Data)
            x_raw = dataset[cluster].values
            y_raw = dataset[subscore].values

            # Rank transform for Spearman correlation
            x_ranked = dataset[cluster].rank().values
            y_ranked = dataset[subscore].rank().values

            # Apply Yeo-Johnson transformation
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            x_yeo = pt.fit_transform(x_raw.reshape(-1, 1)).flatten()
            y_yeo = pt.fit_transform(y_raw.reshape(-1, 1)).flatten()

            # --- STATISTICAL TESTS (NO JITTER) ---
            # Spearman correlation (raw data - ranks)
            spearman_r, spearman_p = spearmanr(x_raw, y_raw)

            # Pearson correlation (Yeo-Johnson transformed data)
            pearson_r_yeo, pearson_p_yeo = pearsonr(x_yeo, y_yeo)

            # Print correlation results
            print(f"{cluster} vs {subscore}: Spearman R = {spearman_r:.2f}, P = {spearman_p:.2e}")
            print(f"{cluster} vs {subscore} (Yeo-Johnson): Pearson R = {pearson_r_yeo:.2f}, P = {pearson_p_yeo:.2e}")

            # --- APPLY JITTER ONLY FOR VISUALIZATION ---
            np.random.seed(50)  # Fix seed for reproducibility

            # Apply the SAME jitter for both raw and Yeo if the specific plot requires it
            if cluster == 'fano_factor_Cluster_1' and subscore == 'all_score':
                # Generate jitter values (same jitter applied to both)
                x_jitter = np.random.uniform(-x_jitter_target, x_jitter_target, len(x_raw))
                y_jitter = np.random.uniform(-y_jitter_target, y_jitter_target, len(y_raw))

                # Apply jitter to raw data
                x_raw_jittered = x_raw + x_jitter
                y_raw_jittered = y_raw + y_jitter
                x_raw_jittered = np.clip(x_raw_jittered, 0, None)  # Ensure no negatives

                # Apply the SAME jitter to Yeo-Johnson transformed data
                x_yeo_jittered = x_yeo + x_jitter
                y_yeo_jittered = y_yeo + y_jitter
                x_yeo_jittered = np.clip(x_yeo_jittered, 0, None)  # Ensure no negatives
            else:
                # No jitter for other plots
                x_raw_jittered = x_raw
                y_raw_jittered = y_raw
                x_yeo_jittered = x_yeo
                y_yeo_jittered = y_yeo

            # --- PLOTTING ---
            fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=False)

            # Subplot 1: Raw Data (with jitter if applicable)
            sns.regplot(x=x_raw_jittered, y=y_raw_jittered, ax=axs[0],
            scatter_kws={'color': 'darkgrey', 'edgecolor': 'dimgrey', 'linewidth': 1.5, 's': 125, 'alpha': 1},
            line_kws={'color': 'tomato', 'linewidth': 4},
            ci=95, lowess=False)  # Linear fit

            axs[0].set_title(f"Raw Data\nSpearman R = {spearman_r:.2f}, P = {spearman_p:.2e}",
                             fontsize=20, weight='bold')
            axs[0].set_xlabel(cluster.replace('_', ' '), fontsize=20, weight='bold')
            axs[0].set_ylabel(subscore.replace('_', ' '), fontsize=20, weight='bold')
            axs[0].tick_params(axis='both', labelsize=20, width=1.2)
            for label in axs[0].get_xticklabels() + axs[0].get_yticklabels():
                label.set_fontweight('bold')
            axs[0].grid(True, linestyle='--', alpha=0.6)

            # Subplot 2: Ranked Data (Trend Visualization)
            sns.regplot(x=x_ranked, y=y_ranked, ax=axs[1],
                        scatter_kws={'color': 'darkgrey', 'edgecolor': 'dimgrey', 'linewidth': 1.5, 's': 125, 'alpha': 1},
                        line_kws={'color': 'tomato', 'linewidth': 5}, ci=95)
            axs[1].set_title(f"Ranked Data\nSpearman R = {spearman_r:.2f}, P = {spearman_p:.2e}",
                             fontsize=20, weight='bold')
            axs[1].set_xlabel(f"Ranked {cluster.replace('_', ' ')}", fontsize=20, weight='bold')
            axs[1].set_ylabel(f"Ranked {subscore.replace('_', ' ')}", fontsize=20, weight='bold')
            axs[1].tick_params(axis='both', labelsize=20, width=1.2)
            for label in axs[1].get_xticklabels() + axs[1].get_yticklabels():
                label.set_fontweight('bold')
            axs[1].grid(True, linestyle='--', alpha=0.6)

            # Subplot 3: Yeo-Johnson Transformed Data
            sns.regplot(x=x_yeo_jittered, y=y_yeo_jittered, ax=axs[2],
                        scatter_kws={'color': 'darkgrey', 'edgecolor': 'dimgrey', 'linewidth': 1.5, 's': 125, 'alpha': 1},
                        line_kws={'color': 'tomato', 'linewidth': 5}, ci=95)
            axs[2].set_title(f"Yeo-Johnson\nPearson R = {pearson_r_yeo:.2f}, P = {pearson_p_yeo:.2e}",
                             fontsize=20, weight='bold')
            axs[2].set_xlabel(f"Yeo-Johnson {cluster.replace('_', ' ')}", fontsize=20, weight='bold')
            axs[2].set_ylabel(f"Yeo-Johnson {subscore.replace('_', ' ')}", fontsize=20, weight='bold')
            axs[2].tick_params(axis='both', labelsize=20, width=1.2)
            for label in axs[2].get_xticklabels() + axs[2].get_yticklabels():
                label.set_fontweight('bold')
            axs[2].grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()
            plt.show()
        else:
            print(f"Missing data for {cluster} or {subscore}. Skipping plot.")


# Call the function
visualize_significant_correlations_OLS_yeo_with_ranks(
    features=features,
    clinical_scores=clinical_scores,
    raw_pvalues=p_values_by_cluster_OLS,
    dataset=final_cellclinics_df_imputed,
    significance_level=0.05,
    x_jitter_target=0.01,
    y_jitter_target=0.02
)


#%% Correlation matrix within a feature and across measures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- generic heatmap (matplotlib-only, no saving) ----------
def plot_corr_heatmap(df, title, vmin=-1.0, vmax=1.0, fmt=".3f"):
    """
    df: pandas DataFrame (rows x cols) of correlations
    title: str
    vmin/vmax: color scale bounds (keep symmetric for correlations)
    fmt: annotation format (e.g., '.2f', '.3f')
    """
    data = df.values.astype(float)
    nrows, ncols = data.shape

    fig, ax = plt.subplots(figsize=(1.2*ncols + 3, 0.6*nrows + 2))
    im = ax.imshow(data, vmin=vmin, vmax=vmax, aspect='auto', origin='upper', cmap='coolwarm')

    # ticks and labels
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels(df.columns, rotation=30, ha='right', fontsize=11, fontweight='bold')
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(df.index, fontsize=11, fontweight='bold')

    # colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation (r)", rotation=90, fontsize=12, fontweight='bold')
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight('bold')

    # annotations
    for i in range(nrows):
        for j in range(ncols):
            ax.text(j, i, format(data[i, j], fmt),
                    ha="center", va="center", fontsize=10, color="black")

    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.show()


# ---------- builders: compute correlations from your DataFrame ----------
def within_feature_corr(
    df: pd.DataFrame,
    feature_name: str,
    clinical_scores: list,
    cluster_count: int = 4,
    method: str = "pearson",   # or "spearman"
) -> pd.DataFrame:
    """
    Rows = clinical scores, Cols = f"{feature_name}_Cluster_i"
    Correlates each symptom with each cluster of the chosen feature.
    """
    cols = [f"{feature_name}_Cluster_{i}" for i in range(1, cluster_count+1)]
    # keep only columns that actually exist
    cols = [c for c in cols if c in df.columns]
    scores = [s for s in clinical_scores if s in df.columns]
    if not cols or not scores:
        raise ValueError("No matching columns found for requested feature/symptoms.")

    # pick correlation function
    func = pd.DataFrame.corrwith
    out = []
    for s in scores:
        if method == "spearman":
            # rank both before corr to mimic Spearman
            tmp = df[cols].rank()
            r = tmp.corrwith(df[s].rank(), method="pearson")
        else:
            r = df[cols].corrwith(df[s], method="pearson")
        out.append(r.values)
    return pd.DataFrame(out, index=scores, columns=cols)


def across_measures_corr(
    df: pd.DataFrame,
    anchor_cluster: str,       # e.g., "fano_factor_Cluster_1"
    all_features: list,        # e.g., ['firing','burst_index','fano_factor','lv','renyi_entropy','mean_isi']
    cluster_count: int = 4,
    method: str = "pearson",   # or "spearman"
) -> pd.DataFrame:
    """
    Row = anchor_cluster, Cols = all other feature clusters (excluding the anchor itself).
    """
    if anchor_cluster not in df.columns:
        raise ValueError(f"{anchor_cluster} not in DataFrame.")
    # build all other cluster columns
    other_cols = []
    for f in all_features:
        for i in range(1, cluster_count+1):
            name = f"{f}_Cluster_{i}"
            if name != anchor_cluster and name in df.columns:
                other_cols.append(name)
    if not other_cols:
        raise ValueError("No other feature-cluster columns found.")

    if method == "spearman":
        r = df[other_cols].rank().corrwith(df[anchor_cluster].rank(), axis=0, method="pearson")
    else:
        r = df[other_cols].corrwith(df[anchor_cluster], axis=0, method="pearson")

    # return as a 1-row DataFrame for easy heatmap plotting
    return pd.DataFrame([r.values], index=[anchor_cluster], columns=other_cols)


# ---------- convenience wrappers to plot (no saving) ----------
def plot_within_feature(df, feature_name, clinical_scores, cluster_count=4, method="spearman"):
    mat = within_feature_corr(df, feature_name, clinical_scores, cluster_count, method)
    title = f"Within-feature correlations — {feature_name.replace('_',' ')} ({method})"
    plot_corr_heatmap(mat, title=title, vmin=-1, vmax=1, fmt=".3f")

def plot_across_measures(df, anchor_cluster, all_features, cluster_count=4, method="spearman"):
    mat = across_measures_corr(df, anchor_cluster, all_features, cluster_count, method)
    title = f"Across-measures correlations — anchor: {anchor_cluster.replace('_',' ')} ({method})"
    plot_corr_heatmap(mat, title=title, vmin=-1, vmax=1, fmt=".3f")

# -------- run ALL features (within-feature + across-measures) --------
# assumes you already defined:
#   - plot_within_feature(...)
#   - plot_across_measures(...)
#   - final_cellclinics_df_imputed  (your DataFrame)
#   - features, clinical_scores     (your lists)
# If not, define those first (same names you’re already using).

def plot_all_correlations(df,
                          features,
                          clinical_scores,
                          cluster_count=4,
                          method="spearman",
                          do_within_feature=True,
                          do_across_measures=True,
                          anchors="all"):  # "all" or list like ["fano_factor_Cluster_1", ...]
    """
    Generates correlation heatmaps for:
      1) Within-feature: symptoms vs clusters for each feature.
      2) Across-measures: anchor cluster vs ALL other feature-clusters.

    anchors:
      - "all": use every feature's clusters (Feature_Cluster_1..k) as anchors
      - list of specific anchor column names to limit the across-measures run
    """
    # 1) Within-feature for every feature
    if do_within_feature:
        for f in features:
            try:
                plot_within_feature(df=df,
                                    feature_name=f,
                                    clinical_scores=clinical_scores,
                                    cluster_count=cluster_count,
                                    method=method)
            except Exception as e:
                print(f"[within-feature] Skipping {f}: {e}")

    # 2) Across-measures
    if do_across_measures:
        # build full anchor list if needed
        if anchors == "all":
            anchor_list = []
            for f in features:
                for i in range(1, cluster_count+1):
                    name = f"{f}_Cluster_{i}"
                    if name in df.columns:
                        anchor_list.append(name)
        else:
            anchor_list = [a for a in anchors if a in df.columns]

        for anchor in anchor_list:
            try:
                plot_across_measures(df=df,
                                     anchor_cluster=anchor,
                                     all_features=features,
                                     cluster_count=cluster_count,
                                     method=method)
            except Exception as e:
                print(f"[across-measures] Skipping {anchor}: {e}")


# ---- call it (everything) ----
plot_all_correlations(
    df=final_cellclinics_df_imputed,
    features=['firing','burst_index','fano_factor','lv','renyi_entropy','mean_isi'],
    clinical_scores=['tremor_score','bradikynesia_score','rigidity_score','axial_score','all_score'],
    cluster_count=4,
    method="spearman",          # or "spearman"
    do_within_feature=True,
    do_across_measures=True,
    anchors="all"              # or e.g. ["fano_factor_Cluster_1","mean_isi_Cluster_4"]
)


#%% VIF

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def vif_table_for_outcome(df, feature_list, outcome, clusters=4):
    """
    Build the X matrix the same way you fit OLS (all clusters of all features), add const, compute VIF.
    Returns a DataFrame sorted by VIF descending.
    """
    cols = []
    for f in feature_list:
        cols += [f"{f}_Cluster_{i}" for i in range(1, clusters+1)]
    # Drop rows with missing data in X or Y
    data = df[[outcome] + cols].dropna().copy()
    X = data[cols]
    X = sm.add_constant(X)

    vif = []
    for i, name in enumerate(X.columns):
        if name.lower() in ("const", "intercept"):
            continue
        vif.append({"predictor": name, "VIF": variance_inflation_factor(X.values, i)})
    vif_df = pd.DataFrame(vif).sort_values("VIF", ascending=False).reset_index(drop=True)
    return vif_df

# Example: VIF for each clinical score (using all features/clusters you include in OLS)
for sub in clinical_scores:
    vif_df = vif_table_for_outcome(final_cellclinics_df_imputed, features, outcome=sub, clusters=4)
    print(f"\nVIF — {sub}")
    print(vif_df.to_string(index=False))


def plot_vif_bars(vif_df, title="Variance Inflation Factors", save_path=None):
    """
    Plot VIFs as a horizontal bar chart.

    Parameters
    ----------
    vif_df : pd.DataFrame
        Must have columns ["predictor", "VIF"].
    title : str
        Figure title.
    save_path : str or None
        If provided, saves the figure to this path.
    """
    # basic checks + sort
    assert {"predictor","VIF"} <= set(vif_df.columns), "vif_df must have ['predictor','VIF']"
    df = vif_df.copy().sort_values("VIF", ascending=True)  # ascending for top-down plot
    preds = df["predictor"].tolist()
    vals  = df["VIF"].astype(float).values

    # color by thresholds: <5 grey, 5–10 orange, ≥10 tomato
    colors = []
    for v in vals:
        if v < 5:
            colors.append("dimgrey")
        elif v < 10:
            colors.append("orange")
        else:
            colors.append("tomato")

    fig_h = max(2.5, 0.45*len(preds) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y = np.arange(len(preds))
    bars = ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.6)

    # annotate each bar with its VIF
    for i, b in enumerate(bars):
        x = b.get_width()
        ax.text(x + max(0.02*x, 0.2), b.get_y() + b.get_height()/2,
                f"{vals[i]:.2f}", va="center", ha="left", fontsize=10, weight="bold")

    # guideline thresholds
    ax.axvline(10, color="black", linestyle=":",  linewidth=1.5, label="VIF=10")

    ax.set_yticks(y)
    ax.set_yticklabels([p.replace("_"," ") for p in preds], fontsize=10, fontweight="bold")
    ax.set_xlabel("Variance Inflation Factor (VIF)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", frameon=False)

    # nice x-lims with headroom for labels
    xmax = max(vals) if len(vals) else 1
    ax.set_xlim(0, xmax*1.25 + 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
plot_vif_bars(vif_df, title="VIF")

