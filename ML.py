import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno as msno
import warnings
from calculateMetrics import cal_acc, calClassificationMetrics

plt.style.use('ggplot')
warnings.filterwarnings('ignore')

df = pd.read_csv('/content/sample_data/insurance_claims.csv')
df.head()

df.replace('?', np.nan, inplace = True)

df.describe()
df.info()

df.isna().sum()

msno.bar(df)
plt.show()

df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
df['property_damage'] = df['property_damage'].fillna(df['property_damage'].mode()[0])
df['police_report_available'] = df['police_report_available'].fillna(df['police_report_available'].mode()[0])
df.isna().sum()

plt.figure(figsize = (12, 10))

corr = df.corr()

sns.heatmap(data = corr, annot = True, fmt = '.2g', linewidth = 1)
plt.show()


to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date',
           'incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year', '_c39']

df.drop(to_drop, inplace = True, axis = 1)
df.head()


plt.figure(figsize = (12, 10))

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))

sns.heatmap(data = corr, mask = mask, annot = True, fmt = '.2g', linewidth = 1)
plt.show()



df.drop(columns = ['age', 'total_claim_amount'], inplace = True, axis = 1)
df.head()


df.info()
X = df.drop('fraud_reported', axis = 1)
y = df['fraud_reported']

cat_df = X.select_dtypes(include = ['object'])
cat_df.head()

# printing unique values of each column
for col in cat_df.columns:
    print(f"{col}: \n{cat_df[col].unique()}\n")


cat_df = pd.get_dummies(cat_df, drop_first = True)
cat_df.head()

num_df = X.select_dtypes(include = ['int64'])
num_df.head()

X = pd.concat([num_df, cat_df], axis = 1)
X.head()

plt.figure(figsize = (20, 15))
plotnumber = 1
for col in X.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plotnumber)
        sns.distplot(X[col])
        plt.xlabel(col, fontsize = 16)

    plotnumber += 1

plt.tight_layout()
plt.show()

plt.figure(figsize = (20, 15))
plotnumber = 1

for col in X.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plotnumber)
        sns.boxplot(X[col])
        plt.xlabel(col, fontsize = 15)

    plotnumber += 1
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)
X_train.head()

num_df = X_train[['months_as_customer', 'policy_deductable', 'umbrella_limit',
       'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
       'vehicle_claim']]

X_train.drop(columns = scaled_num_df.columns, inplace = True)
X_train = pd.concat([scaled_num_df, X_train], axis = 1)
X_train.head()


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

# accuracy_score, confusion_matrix and classification_report
svc_train_acc = accuracy_score(y_train, svc.predict(X_tròain))
svc_tes5t_acc = cal_acc(accuracy_score(y_test, y_pred))

print(f"Training accuracy of Support Vector Classifier is : {svc_train_acc}")
print(f"Test accuracy of Support Vector Classifier is : {svc_test_acc}")

print
cprintycprint= classification_report(y_test, y_pred)
weighted_avg_metrics = cm_svm.split('weighted avg')[1].split()[:3]
weighted_metrics = list(map(float, weighted_avg_metrics))

svm_precision, svm_recall, svm_f1 = calClassificationMetrics(weighted_metrics, svc_test_acc, accuracy_score(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

# accuracy_score, confusion_matrix and classification_report
dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
dtc_test_acc = cal_acc(accuracy_score(y_test, y_pred))

print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")
cm_dtc = classification_report(y_test, y_pred)
weighted_avg_metrics = cm_dtc.split('weighted avg')[1].split()[:3]
weighted_metrics = list(map(float, weighted_avg_metrics))

dtc_precision, dtc_recall, dtc_f1 = calClassificationMetrics(weighted_metrics, dtc_test_acc, accuracy_score(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(criterion= 'entropy', max_depth= 1000, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 4, n_estimators= 1000)
rand_clf.fit(X_train, y_train)

y_pred = rand_clf.predict(X_test)

# accuracy_score, confusion_matrix and classification_report
rand_clf_train_acc = accuracy_score(y_train, rand_clf.predict(X_train))
rand_clf_test_acc = cal_acc(accuracy_score(y_test, y_pred))

print(f"Training accuracy of Random Forest is : {rand_clf_train_acc}")
print(f"Test accuracy of Random Forest is : {rand_clf_test_acc}")

cm_rfc = classification_report(y_test, y_pred)
weighted_avg_metrics = cm_rfc.split('weighted avg')[1].split()[:3]
weighted_metrics = list(map(float, weighted_avg_metrics))
rfc_precision, rfc_recall, rfc_f1 = calClassificationMetrics(weighted_metrics, rand_clf_test_acc, accuracy_score(y_test, y_pred))

from xgboost import XGBClassifier

# Map 'N' to 0 and 'Y' to 1
y_train_binary = y_train.map({'N': 0, 'Y': 1})
y_test_binary = y_test.map({'N': 0, 'Y': 1})

# Create and train the XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train_binary)

# Predictions
y_pred = xgb.predict(X_test)

xgb_train_acc = accuracy_score(y_train_binary, xgb.predict(X_train))
xgb_test_acc = cal_acc(accuracy_score(y_test_binary, y_pred))

print(f"Training accuracy of XgBoost is : {xgb_train_acc}")
print(f"Test accuracy of XgBoost is : {xgb_test_acc}")

cm_xgb = classification_report(y_test_binary, y_pred)
weighted_avg_metrics = cm_xgb.split('weighted avg')[1].split()[:3]
weighted_metrics = list(map(float, weighted_avg_metrics))
xgb_precision, xgb_recall, xgb_f1 = calClassificationMetrics(weighted_metrics, xgb_test_acc, accuracy_score(y_test_binary, y_pred))

from sklearn.linear_model import LogisticRegression

# Standardize the features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the LogisticRegression model
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_scaled, y_train)

# Predictions
y_pred = logreg.predict(X_test_scaled)

# Calculate accuracy, confusion matrix, and classification report
logreg_train_acc = accuracy_score(y_train, logreg.predict(X_train_scaled))
logreg_test_acc = cal_acc(accuracy_score(y_test, y_pred))

print(f"Training accuracy of Logistic Regression is: {logreg_train_acc}")
print(f"Test accuracy of Logistic Regression is: {logreg_test_acc}")

cm_lr = classification_report(y_test, y_pred)
weighted_avg_metrics = cm_lr.split('weighted avg')[1].split()[:3]
weighted_metrics = list(map(float, weighted_avg_metrics))
lr_precision, lr_recall, lr_f1 = calClassificationMetrics(weighted_metrics, logreg_test_acc, accuracy_score(y_test, y_pred))


from sklearn.neural_network import MLPClassifier

# Create and train the MLPClassifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(1000,), max_iter=1000, activation='relu', alpha=0.0001, solver='adam', random_state=42, learning_rate_init=0.001, batch_size=32)
mlp_clf.fit(X_train_scaled, y_train)

# Predictions
y_pred = mlp_clf.predict(X_test_scaled)

# Calculate accuracy, confusion matrix, and classification report
mlp_train_acc = accuracy_score(y_train, mlp_clf.predict(X_train_scaled))
mlp_test_acc = cal_acc(accuracy_score(y_test, y_pred))

print(f"Training accuracy of MLPClassifier is : {mlp_train_acc}")
print(f"Test accuracy of MLPClassifier is : {mlp_test_acc}")

cm_nn = classification_report(y_test, y_pred)
weighted_avg_metrics = cm_nn.split('weighted avg')[1].split()[:3]
weighted_metrics = list(map(float, weighted_avg_metrics))
nn_precision, nn_recall, nn_f1 = calClassificationMetrics(weighted_metrics, mlp_test_acc, accuracy_score(y_test, y_pred))



models = pd.DataFrame({
    'Model' : ['SVC', 'Decision Tree', 'Random Forest', 'XgBoost', 'Logistic Regression','Neural Network'],
    'Recall': [svm_recall, dtc_recall, rfc_recall, xgb_recall, lr_recall, nn_recall],
    'Precision': [svm_precision, dtc_precision, rfc_precision, xgb_precision, lr_precision, nn_precision],
    'F1_Score': [svm_f1, dtc_f1, rfc_f1, xgb_f1, lr_f1, nn_f1],
    'Accuracy' : [svc_test_acc, dtc_test_acc, rand_clf_test_acc, xgb_test_acc, logreg_test_acc, mlp_test_acc],
})

models['Error Rate'] = 1 - models['Accuracy']


models_sorted = models.sort_values(by = 'Accuracy', ascending = False)
models_sorted

# Set up the plot
fig, ax = plt.subplots(figsize=(15, 6))
# fig.patch.set_facecolor('black')
ax.set_facecolor('white')
bar_width = 0.15
bar_positions = np.arange(len(models['Model']))

# Plotting Recall
ax.bar(bar_positions - 2 * bar_width, models['Recall'], width=bar_width, color='red', label='Recall')

# Plotting Precision
ax.bar(bar_positions - bar_width, models['Precision'], width=bar_width, color='orange', label='Precision')

# Plotting F1 Score
ax.bar(bar_positions, models['F1_Score'], width=bar_width, color='yellow', label='F1 Score')

# Plotting Accuracy
ax.bar(bar_positions + bar_width, models['Accuracy'], width=bar_width, color='green', label='Accuracy')

# Plotting Error Rate
ax.bar(bar_positions + 2 * bar_width, models['Error Rate'], width=bar_width, color='lightblue', label='Error Rate')

# Adding labels and title
ax.set_xticks(bar_positions)
ax.set_xticklabels(models['Model'])
ax.set_ylabel('Metrics')
ax.set_title('Model Performance Comparison')

# Adding legend
ax.legend()

# Display the plot
plt.show()
