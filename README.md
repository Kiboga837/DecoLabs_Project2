# DecoLabs_Project2
Supervised Learning: Fraud Detection Pipeline (Leak-Free, SMOTE + Imblearn)
### Process Highlight
1. Loading the dataset and builds a binary target (see TARGET SETUP below).
2. Showing the class imbalance and the "Accuracy Illusion" (dummy baseline).
3. Doing a STRATIFIED 80/20 train/test split BEFORE any scaling / resampling.
4. Building two imblearn Pipelines:
       Logistic Regression : StandardScaler -> SMOTE -> LogisticRegression
       Random Forest       : SMOTE -> RandomForestClassifier
5. Tunning preprocessing + model hyperparameters together with GridSearchCV
   (5-fold stratified CV, SMOTE applied only inside each training fold).
6. Evaluating on the untouched test set with Precision, Recall, F1, ROC-AUC,
   PR-AUC and confusion matrices. Accuracy is deliberately NOT used.
7. Runnning an ablation (Random Forest with vs. without SMOTE), tunes the decision
   threshold using out-of-fold predictions only, and saves all artefacts.
