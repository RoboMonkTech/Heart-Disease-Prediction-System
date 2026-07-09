# ==========================================
# HEART DISEASE PREDICTION SYSTEM
# Professional Model Comparison Pipeline
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("dataset/heart_disease.csv")

X = df.drop(["num","target_binary"],axis=1)

y = df["target_binary"]

# ==========================================
# Split Dataset
# ==========================================

X_train,X_test,y_train,y_test=train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

# ==========================================
# Models
# ==========================================

models={

"Logistic Regression":(

LogisticRegression(max_iter=3000),

{

"model__C":[0.01,0.1,1,10],

"model__solver":["lbfgs","liblinear"]

}

),

"Random Forest":(

RandomForestClassifier(random_state=42),

{

"model__n_estimators":[200,300],

"model__max_depth":[10,15,None]

}

),

"Extra Trees":(

ExtraTreesClassifier(random_state=42),

{

"model__n_estimators":[200,300],

"model__max_depth":[10,15,None]

}

),

"Gradient Boosting":(

GradientBoostingClassifier(random_state=42),

{

"model__learning_rate":[0.05,0.1],

"model__n_estimators":[100,200]

}

)

}

# ==========================================
# Compare Models
# ==========================================

results=[]

best_pipeline=None

best_accuracy=0

best_name=""

print("="*70)

print("MODEL COMPARISON")

print("="*70)

for name,(model,param_grid) in models.items():

    pipeline=Pipeline([

        ("scaler",StandardScaler()),

        ("model",model)

    ])

    grid=GridSearchCV(

        pipeline,

        param_grid,

        cv=cv,

        scoring="accuracy",

        n_jobs=-1

    )

    grid.fit(X_train,y_train)

    best=grid.best_estimator_

    prediction=best.predict(X_test)

    probability=best.predict_proba(X_test)[:,1]

    accuracy=accuracy_score(y_test,prediction)

    precision=precision_score(y_test,prediction)

    recall=recall_score(y_test,prediction)

    f1=f1_score(y_test,prediction)

    auc=roc_auc_score(y_test,probability)

    results.append({

        "Model":name,

        "CV Accuracy":grid.best_score_,

        "Test Accuracy":accuracy,

        "Precision":precision,

        "Recall":recall,

        "F1":f1,

        "ROC AUC":auc

    })

    if accuracy>best_accuracy:

        best_accuracy=accuracy

        best_pipeline=best

        best_name=name

# ==========================================
# Results
# ==========================================

results=pd.DataFrame(results)

results=results.sort_values(

    by="Test Accuracy",

    ascending=False

)

print(results)

print("\n")

print("="*70)

print("BEST MODEL")

print("="*70)

print(best_name)

print("\nAccuracy :",round(best_accuracy*100,2),"%")

joblib.dump(

    best_pipeline,

    "model/heart_pipeline.pkl"

)

print("\nBest Model Saved Successfully")