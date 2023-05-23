import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn import metrics
data = pd.read_csv(r'C:\Users\denis\OneDrive\Desktop\Python\Folio_project\train_data.csv')

y = data['PRICE']
data = data.drop('PRICE', axis=1)

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=42)



# Saving the index
X_train['original_index'] = X_train.index
X_val['original_index'] = X_val.index

# imputing BUILD_YEAR based on the average of each SUBURB
imputer = SimpleImputer()

def impute_group(group):
    group['BUILD_YEAR'] = imputer.fit_transform(group[['BUILD_YEAR']])
    return group

X_train = X_train.groupby('SUBURB').apply(impute_group)
X_val = X_val.groupby('SUBURB').apply(impute_group)

# Imputing remaining missing values with the median BUILD_YEAR across the entire dataset
if X_train['BUILD_YEAR'].isnull().sum() > 0:
    X_train['BUILD_YEAR'].fillna(X_train['BUILD_YEAR'].median(), inplace=True)

if X_val['BUILD_YEAR'].isnull().sum() > 0:
    X_val['BUILD_YEAR'].fillna(X_val['BUILD_YEAR'].median(), inplace=True)

X_train = X_train.reset_index(drop=True)
X_train = X_train.set_index('original_index')
X_train = X_train.sort_index()

X_val = X_val.reset_index(drop=True)
X_val = X_val.set_index('original_index')
X_val = X_val.sort_index()

# Imputing garage to '0'
X_train['GARAGE'].fillna(0, inplace=True)
X_val['GARAGE'].fillna(0, inplace=True)




num_cols = [cname for cname in data.columns if data[cname].dtype in ['int64', 'float64']]

# correlation_matrix = data[num_cols].corr()
# print(correlation_matrix['NEAREST_SCH_RANK'].sort_values(ascending=False))

total_houses = data['SUBURB'].value_counts()
missing_rank = data['NEAREST_SCH_RANK'].isnull().groupby(data['SUBURB']).sum().astype(int)
missing_rank_proportion = missing_rank / total_houses

subs_all_missing_rank = missing_rank_proportion[missing_rank_proportion == 1]

# Print number of such suburbs and their names
print("Number of suburbs with all houses missing a school rank:", len(subs_all_missing_rank))
print("Suburbs:", ', '.join(subs_all_missing_rank.index))
# subset of original dataframe with rows from suburbs that have 100% missing 'NEAREST_SCH_RANK'
data_subs_all_missing = data[data['SUBURB'].isin(subs_all_missing_rank.index)]

# How many houses are in subs with all the houses missing school rank?
num_houses_subs_all_missing = len(data_subs_all_missing)
total_houses_missing = data['NEAREST_SCH_RANK'].isnull().sum()

percentage = (num_houses_subs_all_missing / total_houses_missing) * 100
print("Percentage of houses with missing 'NEAREST_SCH_RANK' that are in suburbs with 100% missing rank: {:.2f}%".format(percentage))

# Calculate means for each suburb in the training set
suburb_means = X_train.groupby('SUBURB')['NEAREST_SCH_RANK'].mean()

# Impute missing values in training data
mask_all_missing_train = X_train['SUBURB'].isin(subs_all_missing_rank.index)
X_train.loc[mask_all_missing_train, 'NEAREST_SCH_RANK'] = 0
X_train.loc[~mask_all_missing_train, 'NEAREST_SCH_RANK'] = X_train.loc[~mask_all_missing_train].groupby('SUBURB')['NEAREST_SCH_RANK'].transform(lambda x: x.fillna(x.mean()))

# Fill remaining NaNs in the training set (if any) with 0 or global mean
X_train['NEAREST_SCH_RANK'].fillna(0, inplace=True)

# Impute missing values in validation data based on the means from the training data
mask_all_missing_val = X_val['SUBURB'].isin(subs_all_missing_rank.index)
X_val.loc[mask_all_missing_val, 'NEAREST_SCH_RANK'] = 0
X_val['NEAREST_SCH_RANK'] = X_val['SUBURB'].map(suburb_means)  # Map means of 'NEAREST_SCH_RANK' based on 'SUBURB' in validation data

# If there are any suburbs in the validation set that were not present in the training set, 'NEAREST_SCH_RANK' would still be NaN for these suburbs
# Fill these remaining NaNs with 0 or some other appropriate value
X_val['NEAREST_SCH_RANK'].fillna(0, inplace=True)



# Dealing with categorical columns..
# Very sophisticated method..
# I'm about to show you..

cat_columns = [cname for cname in data.columns if data[cname].dtype == 'object']

X_train.drop(cat_columns, axis =1, inplace=True)
X_val.drop(cat_columns, axis =1, inplace=True)


# Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)

y_scaler = StandardScaler()
# We're using reshape(-1, 1) because the scaler requires a 2D array as input
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1))

# Because the output of the scaler is a 2D array, we'll also flatten() the scaled y to convert it back to 1D
y_train_scaled = y_train_scaled.flatten()
y_val_scaled = y_val_scaled.flatten()



# create a SVR object
svr = svm.SVR()

svr.fit(X_train_scaled, y_train_scaled)

# predict the target for the validation set
y_val_pred = svr.predict(X_val_scaled)

# Transform the prediction back to the original scale
y_val_pred_orig = y_scaler.inverse_transform(y_val_pred.reshape(-1, 1))
y_val_descaled = y_scaler.inverse_transform(y_val_scaled.reshape(-1, 1))

mae = mean_absolute_error(y_val_descaled, y_val_pred_orig)
print('Mean Negative Error on validation set: ', mae)
