import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math


players = pd.read_csv("merged_training_table.csv")

target_col = "rookie_per" 
X = players.drop(columns=[target_col])
X = X.dropna(axis=1, how='all')
y = players[target_col]

# Selecting columns
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# models
models = {
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, objective="reg:squarederror")
}

# baseline stuff
cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    cv_results = cross_validate(pipe, X, y, cv=cv,
                                scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"])
    results[name] = {
        "MAE": -np.mean(cv_results["test_neg_mean_absolute_error"]),
        "RMSE": -np.mean(cv_results["test_neg_root_mean_squared_error"]),
        "R2": np.mean(cv_results["test_r2"])
    }

print("Baseline CV Results:")
print(pd.DataFrame(results).T)

# changing hyperparameters
grids = {
    "DecisionTree": {"model__max_depth": [2, 3, 4, 6, 8],
                     "model__min_samples_split": [2, 5, 10]},
    "RandomForest": {"model__n_estimators": [100, 200],
                     "model__max_depth": [6, 8, 10],
                     "model__min_samples_split": [2, 5]},
    "XGBoost": {"model__n_estimators": [100, 200],
                "model__max_depth": [3, 4, 5],
                "model__learning_rate": [0.05, 0.1]}
}

tuned_results = {}
best_params = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    grid = GridSearchCV(pipe, grids[name],
                        scoring="neg_mean_absolute_error",
                        cv=cv, n_jobs=-1)
    grid.fit(X, y)
    best_params[name] = grid.best_params_
    best_model = grid.best_estimator_
    
    y_pred = grid.predict(X)
    tuned_results[name] = {
        "MAE": mean_absolute_error(y, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y, y_pred)),
        "R2": r2_score(y, y_pred)
    }

print("\nTuned Model Results:")
print(pd.DataFrame(tuned_results).T)
print("\nBest Hyperparameters:")
print(best_params)

# feature importance
def get_feature_names():
    ohe = preprocessor.named_transformers_["cat"]["encoder"]
    cat_names = ohe.get_feature_names_out(cat_cols)
    return num_cols + list(cat_names)

feature_importances = {}
for name, model in models.items():
    pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    pipe.fit(X, y)
    feat_names = get_feature_names()
    try:
        importances = pipe.named_steps["model"].feature_importances_
        feature_importances[name] = pd.DataFrame({
            "feature": feat_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        feature_importances[name].to_csv(f"feature_importance_{name}.csv", index=False)
    except AttributeError:
        print(f"{name} does not support feature_importances_")

# IG and gain ratio
def entropy(series):
    probs = series.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-9))

def info_gain(parent, left, right):
    H_parent = entropy(parent)
    n = len(parent)
    n_l, n_r = len(left), len(right)
    return H_parent - (n_l/n)*entropy(left) - (n_r/n)*entropy(right)

def gain_ratio(parent, left, right):
    ig = info_gain(parent, left, right)
    split_info = entropy(pd.Series(["L"]*len(left) + ["R"]*len(right)))
    return ig / (split_info + 1e-9)

tree = DecisionTreeRegressor(max_depth=best_params["DecisionTree"]["model__max_depth"],
                             min_samples_split=best_params["DecisionTree"]["model__min_samples_split"],
                             random_state=42)
pipe = Pipeline(steps=[("pre", preprocessor), ("model", tree)])
pipe.fit(X, y)

feat_names = get_feature_names()
dt = pipe.named_steps["model"]
X_proc = pipe.named_steps["pre"].transform(X)
X_proc = X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc
y_bins = pd.cut(y, bins=10)

gain_rows = []
for node in range(dt.tree_.node_count):
    if dt.tree_.children_left[node] != -1:  # split node
        feat_idx = dt.tree_.feature[node]
        thr = dt.tree_.threshold[node]
        feat_name = feat_names[feat_idx]
        left_idx = X_proc[:, feat_idx] <= thr
        right_idx = X_proc[:, feat_idx] > thr
        gain = info_gain(y_bins, y_bins[left_idx], y_bins[right_idx])
        ratio = gain_ratio(y_bins, y_bins[left_idx], y_bins[right_idx])
        gain_rows.append((feat_name, thr, gain, ratio))

gain_df = pd.DataFrame(gain_rows, columns=["feature", "threshold", "info_gain", "gain_ratio"])
gain_df.to_csv("decision_tree_gain_ratio.csv", index=False)

# summary
summary = pd.concat({
    "Untuned": pd.DataFrame(results).T,
    "Tuned": pd.DataFrame(tuned_results).T
}, axis=1)
summary.to_csv("model_comparison_summary.csv")
print("\nSaved: model_comparison_summary.csv, decision_tree_gain_ratio.csv, feature_importance_*.csv")