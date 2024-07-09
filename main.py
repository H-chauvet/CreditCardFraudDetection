import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

df = pd.read_csv('creditcard.csv')

scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'criterion': ['gini', 'entropy']
}

clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)

y_pred_optimized = best_clf.predict(X_test)
y_proba_optimized = best_clf.predict_proba(X_test)[:, 1]

print("Optimized Accuracy:", accuracy_score(y_test, y_pred_optimized))
print("Optimized ROC AUC Score:", roc_auc_score(y_test, y_proba_optimized))
print("Optimized Classification Report:\n", classification_report(y_test, y_pred_optimized))

fpr, tpr, thresholds = roc_curve(y_test, y_proba_optimized)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

results = pd.DataFrame({
    'Transaction': X_test.index,
    'Predicted Class': y_pred_optimized,
    'Actual Class': y_test,
    'Probability of Fraud': y_proba_optimized
})

results['Fraudulent'] = results['Predicted Class'] == 1
results['Reason'] = results['Probability of Fraud'].apply(lambda x: 'High probability' if x > 0.5 else 'Low probability')

fraudulent_transactions = results[results['Fraudulent']]
print("Detailed information of predicted fraudulent transactions:")
print(fraudulent_transactions)
