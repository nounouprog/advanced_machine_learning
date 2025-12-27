#!/usr/bin/env python
# coding: utf-8

# <p align="center">
#   <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Logo-gustave-roussy.jpg/1200px-Logo-gustave-roussy.jpg" alt="Logo 1" width="250"/>
#   <img src="https://upload.wikimedia.org/wikipedia/en/thumb/3/3f/Qube_Research_%26_Technologies_Logo.svg/1200px-Qube_Research_%26_Technologies_Logo.svg.png" alt="Logo 2" width="200" style="margin-left: 20px;"/>
# </p>
# 
# # Data Challenge : Leukemia Risk Prediction
# 

# *GOAL OF THE CHALLENGE and WHY IT IS IMPORTANT:*
# 
# The goal of the challenge is to **predict disease risk for patients with blood cancer**, in the context of specific subtypes of adult myeloid leukemias.
# 
# The risk is measured through the **overall survival** of patients, i.e. the duration of survival from the diagnosis of the blood cancer to the time of death or last follow-up.
# 
# Estimating the prognosis of patients is critical for an optimal clinical management. 
# For exemple, patients with low risk-disease will be offered supportive care to improve blood counts and quality of life, while patients with high-risk disease will be considered for hematopoietic stem cell transplantion.
# 
# The performance metric used in the challenge is the **IPCW-C-Index**.

# *THE DATASETS*
# 
# The **training set is made of 3,323 patients**.
# 
# The **test set is made of 1,193 patients**.
# 
# For each patient, you have acces to CLINICAL data and MOLECULAR data.
# 
# The details of the data are as follows:

# - OUTCOME:
#   * OS_YEARS = Overall survival time in years
#   * OS_STATUS = 1 (death) , 0 (alive at the last follow-up)

# - CLINICAL DATA, with one line per patient:
#   
#   * ID = unique identifier per patient
#   * CENTER = clinical center
#   * BM_BLAST = Bone marrow blasts in % (blasts are abnormal blood cells)
#   * WBC = White Blood Cell count in Giga/L 
#   * ANC = Absolute Neutrophil count in Giga/L
#   * MONOCYTES = Monocyte count in Giga/L
#   * HB = Hemoglobin in g/dL
#   * PLT = Platelets coutn in Giga/L
#   * CYTOGENETICS = A description of the karyotype observed in the blood cells of the patients, measured by a cytogeneticist. Cytogenetics is the science of chromosomes. A karyotype is performed from the blood tumoral cells. The convention for notation is ISCN (https://en.wikipedia.org/wiki/International_System_for_Human_Cytogenomic_Nomenclature). Cytogenetic notation are: https://en.wikipedia.org/wiki/Cytogenetic_notation. Note that a karyotype can be normal or abnornal. The notation 46,XX denotes a normal karyotype in females (23 pairs of chromosomes including 2 chromosomes X) and 46,XY in males (23 pairs of chromosomes inclusing 1 chromosme X and 1 chromsome Y). A common abnormality in the blood cancerous cells might be for exemple a loss of chromosome 7 (monosomy 7, or -7), which is typically asssociated with higher risk disease

# - GENE MOLECULAR DATA, with one line per patient per somatic mutation. Mutations are detected from the sequencing of the blood tumoral cells. 
# We call somatic (= acquired) mutations the mutations that are found in the tumoral cells but not in other cells of the body.
# 
#   * ID = unique identifier per patient
#   * CHR START END = position of the mutation on the human genome
#   * REF ALT = reference and alternate (=mutant) nucleotide
#   * GENE = the affected gene
#   * PROTEIN_CHANGE = the consequence of the mutation on the protei that is expressed by a given gene
#   * EFFECT = a broad categorization of the mutation consequences on a given gene.
#   * VAF = Variant Allele Fraction = it represents the **proportion** of cells with the deleterious mutations. 

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored , concordance_index_ipcw
from sklearn.impute import SimpleImputer
from sksurv.util import Surv

# Clinical Data
df = pd.read_csv("./clinical_train.csv")
df_eval = pd.read_csv("./clinical_test.csv")

# Molecular Data
maf_df = pd.read_csv("./molecular_train.csv")
maf_eval = pd.read_csv("./molecular_test.csv")

target_df = pd.read_csv("./target_train.csv")
target_df_test = pd.read_csv("./target_test.csv")

# Preview the data
df.head()


# ### Step 1: Data Preparation (clinical data only)
# 
# For survival analysis, we’ll format the dataset so that OS_YEARS represents the time variable and OS_STATUS represents the event indicator.

# In[3]:


# Define features for the model
features = ['BM_BLAST', 'HB', 'PLT']

# Clean and format the target dataframe
target_df['OS_YEARS'] = pd.to_numeric(target_df['OS_YEARS'], errors='coerce')
target_df['OS_STATUS'] = target_df['OS_STATUS'].astype(bool)

# Drop rows where survival data is missing
target_df.dropna(subset=['OS_YEARS', 'OS_STATUS'], inplace=True)

# Verify formatting
print(target_df[['OS_YEARS', 'OS_STATUS']].dtypes)

# Create the survival data format
X = df.loc[df['ID'].isin(target_df['ID']), features]
y = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', target_df)


# ### Step 2: Splitting the Dataset
# We’ll split the data into training and testing sets to evaluate the model’s performance.

# In[4]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[5]:


# Survival-aware imputation for missing values
imputer = SimpleImputer(strategy="median")
X_train[['BM_BLAST', 'HB', 'PLT']] = imputer.fit_transform(X_train[['BM_BLAST', 'HB', 'PLT']])
X_test[['BM_BLAST', 'HB', 'PLT']] = imputer.transform(X_test[['BM_BLAST', 'HB', 'PLT']])


# ### Step 3: Training Standard Machine Learning Methods
# 
# In this step, we train a standard LightGBM model on survival data, but we do not account for censoring. Instead of treating the event status, we use only the observed survival times as the target variable. This approach disregards whether an individual’s event (e.g., death) was observed or censored, effectively treating the problem as a standard regression task. While this method provides a basic benchmark, it may be less accurate than survival-specific models (but still be explored!), as it does not leverage the information contained in censored observations.

# In[6]:


# Import necessary libraries
import lightgbm as lgb
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# Define LightGBM parameters
lgbm_params = {
    'max_depth': 3,
    'learning_rate': 0.05,
    'verbose': -1
}

# Prepare the data for LightGBM
# Scale the target (OS_YEARS) to reduce skew, apply weights based on event status
X_train_lgb = X_train  # Features for training
y_train_transformed = y_train['OS_YEARS']

# Create LightGBM dataset
train_dataset = lgb.Dataset(X_train_lgb, label=y_train_transformed)

# Train the LightGBM model
model = lgb.train(params=lgbm_params, train_set=train_dataset)

# Make predictions on the training and testing sets
pred_train = -model.predict(X_train)
pred_test = -model.predict(X_test)

# Evaluate the model using Concordance Index IPCW
train_ci_ipcw = concordance_index_ipcw(y_train, y_train, pred_train, tau=7)[0]
test_ci_ipcw = concordance_index_ipcw(y_train, y_test, pred_test, tau=7)[0]
print(f"LightGBM Survival Model Concordance Index IPCW on train: {train_ci_ipcw:.2f}")
print(f"LightGBM Survival Model Concordance Index IPCW on test: {test_ci_ipcw:.2f}")


# In[7]:


# Assuming the LightGBM model is defined as `model`
plt.figure(figsize=(20, 10))
lgb.plot_tree(model, tree_index=0, figsize=(20, 10), show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
plt.title("First Tree in LightGBM Model")
plt.show()


# ### Step 4: Cox Proportional Hazards Model
# 
# To account for censoring in survival analysis, we use a Cox Proportional Hazards (Cox PH) model, a widely used method that estimates the effect of covariates on survival times without assuming a specific baseline survival distribution. The Cox PH model is based on the hazard function, $h(t | X)$, which represents the instantaneous risk of an event (e.g., death) at time $t$ given covariates $X$. The model assumes that the hazard can be expressed as:
# 
# $$h(t | X) = h_0(t) \exp(\beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p)$$
# 
# 
# where $h_0(t)$ is the baseline hazard function, and $\beta$ values are coefficients for each covariate, representing the effect of $X$ on the hazard. Importantly, the proportional hazards assumption implies that the hazard ratios between individuals are constant over time. This approach effectively leverages both observed and censored survival times, making it a more suitable method for survival data compared to standard regression techniques that ignore censoring.
# 

# In[8]:


# Initialize and train the Cox Proportional Hazards model
cox = CoxPHSurvivalAnalysis()
cox.fit(X_train, y_train)

# Evaluate the model using Concordance Index IPCW
cox_cindex_train = concordance_index_ipcw(y_train, y_train, cox.predict(X_train), tau=7)[0]
cox_cindex_test = concordance_index_ipcw(y_train, y_test, cox.predict(X_test), tau=7)[0]
print(f"Cox Proportional Hazard Model Concordance Index IPCW on train: {cox_cindex_train:.2f}")
print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {cox_cindex_test:.2f}")


# ### Step 5: Incorporate Clinical and Molecular Features
# 
# We will now add more clinical features and some aggregated molecular features.

# In[9]:


# 1. Molecular Features
# Calculate Max and Mean VAF per patient
vaf_stats = maf_df.groupby('ID')['VAF'].agg(['max', 'mean']).reset_index()
vaf_stats.rename(columns={'max': 'Max_VAF', 'mean': 'Mean_VAF'}, inplace=True)

# Count mutations per patient (as before)
n_mut = maf_df.groupby('ID').size().reset_index(name='Nmut')

# Binary features for specific genes
# Let's pick a few common genes based on domain knowledge or frequency
top_genes = ['TET2', 'DNMT3A', 'ASXL1', 'FLT3', 'NPM1'] 
gene_features = []

for gene in top_genes:
    # multiple entries per patient possible, so we group by ID
    has_gene = maf_df[maf_df['GENE'] == gene].groupby('ID').size().reset_index(name=f'has_{gene}')
    has_gene[f'has_{gene}'] = (has_gene[f'has_{gene}'] > 0).astype(int)
    gene_features.append(has_gene)

# Merge everything into a molecular stats dataframe
michel_df = n_mut
michel_df = michel_df.merge(vaf_stats, on='ID', how='outer')

for gf in gene_features:
    michel_df = michel_df.merge(gf, on='ID', how='outer')

# Fill NaNs for molecular features (if a patient is not in maf_df, they have 0 mutations/VAF)
# For VAF, if no mutation, VAF is 0.
michel_df.fillna(0, inplace=True)

# Merge with Clinical Data
# New clinical features to include
clinical_features_to_use = ['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES']

df_2 = df.merge(michel_df, on='ID', how='left')

# Fill NaNs for the molecular features we just merged (patients with no mutations)
mol_cols = ['Nmut', 'Max_VAF', 'Mean_VAF'] + [f'has_{g}' for g in top_genes]
df_2[mol_cols] = df_2[mol_cols].fillna(0)


# In[10]:


# Select features
features = clinical_features_to_use + mol_cols
target = ['OS_YEARS', 'OS_STATUS']

# Create the survival data format
X = df_2.loc[df_2['ID'].isin(target_df['ID']), features]
y = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', target_df)


# In[11]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[12]:


# Survival-aware imputation for missing values in CLINICAL data
# Molecular data NaNs were already handled (assumed 0 if not present)
# But standard scaler or imputer might be needed for clinical missing values
imputer = SimpleImputer(strategy="median")

# Fit on train
X_train[features] = imputer.fit_transform(X_train[features])
# Transform test
X_test[features] = imputer.transform(X_test[features])


# In[13]:


# Initialize and train the Cox Proportional Hazards model
cox = CoxPHSurvivalAnalysis()
cox.fit(X_train, y_train)

# Evaluate the model using Concordance Index IPCW
cox_cindex_train = concordance_index_ipcw(y_train, y_train, cox.predict(X_train), tau=7)[0]
cox_cindex_test = concordance_index_ipcw(y_train, y_test, cox.predict(X_test), tau=7)[0]
print(f"Cox Proportional Hazard Model Concordance Index IPCW on train: {cox_cindex_train:.2f}")
print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {cox_cindex_test:.2f}")


# ### Inference on test set

# In[14]:


# Prepare test set molecular stats
vaf_stats_eval = maf_eval.groupby('ID')['VAF'].agg(['max', 'mean']).reset_index()
vaf_stats_eval.rename(columns={'max': 'Max_VAF', 'mean': 'Mean_VAF'}, inplace=True)

n_mut_eval = maf_eval.groupby('ID').size().reset_index(name='Nmut')

gene_features_eval = []
for gene in top_genes:
    has_gene = maf_eval[maf_eval['GENE'] == gene].groupby('ID').size().reset_index(name=f'has_{gene}')
    has_gene[f'has_{gene}'] = (has_gene[f'has_{gene}'] > 0).astype(int)
    gene_features_eval.append(has_gene)

michel_df_eval = n_mut_eval
michel_df_eval = michel_df_eval.merge(vaf_stats_eval, on='ID', how='outer')
for gf in gene_features_eval:
    michel_df_eval = michel_df_eval.merge(gf, on='ID', how='outer')
    
michel_df_eval.fillna(0, inplace=True)

# Merge with clinical test set
df_eval_final = df_eval.merge(michel_df_eval, on='ID', how='left')
df_eval_final[mol_cols] = df_eval_final[mol_cols].fillna(0)


# In[15]:


# Impute missing values in clinical features using the SAME imputer fitted on train
df_eval_final[features] = imputer.transform(df_eval_final[features])

prediction_on_test_set = cox.predict(df_eval_final.loc[:, features])


# In[16]:


prediction_on_test_set


# In[17]:


submission = pd.Series(prediction_on_test_set, index=df_eval['ID'], name='OS_YEARS')


# In[18]:


submission


# In[19]:


submission.to_csv('./benchmark_submission.csv')


# In[20]:


submission


# In[21]:


random_submission = pd.Series(np.random.uniform(0, 1, len(submission)),index =submission.index, name='OS_YEARS')


# In[22]:


random_submission.to_csv('./random_submission.csv')


# In[23]:


random_submission


# In[ ]:




