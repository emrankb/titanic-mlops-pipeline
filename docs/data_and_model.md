# Data preparation and classification

The supplied Titanic CSV has 891 passengers and a binary survival outcome. This project predicts Titanic survival, not hospital readmission. Readmission labels are not present.

The model excludes passenger ID, name, ticket, and cabin from the input features. Cabin has 687 missing values and is not used for prediction. The seven raw features are passenger class, sex, age, family members aboard, fare, and embarkation port.

Feature engineering adds family size, a traveling-alone indicator and fare per family member. The numeric pipeline clips outliers using thresholds fitted on the training fold, imputes missing numeric values and robustly scales the results. The categorical pipeline imputes and one-hot encodes class, sex and port.

The dataset is split 80/20 with stratification. Three-fold stratified cross-validation on the training portion compares a majority baseline, logistic regression and random forest. The highest average cross-validated F1 wins. The final model is evaluated once on the untouched holdout using accuracy, precision, recall, F1 and ROC-AUC.

Run `python -m scripts.train --csv data/raw/titanic.csv` locally. Model files and aggregate results stay in ignored artifact and report directories. The CI tests use fictional rows only.

Do not load untrusted joblib model files, because their pickle-backed representation can execute code.
