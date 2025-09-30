# app.py
# Streamlit interactive "what‑if" plot for a regression model (Random Forest or XGBoost)
# - Auto-build sliders for numeric features and dropdowns for categoricals
# - Lets you choose target + predictors, train model, and vary one feature to see predictions

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Try to import XGBoost; if missing, we will disable that option
try:
    from xgboost import XGBRegressor  # type: ignore
    XGB_OK = True
except Exception:
    XGB_OK = False

# ---------------- Config ----------------
DEFAULT_DATA_PATH = "/Users/Mehr/Downloads/daft.csv"  # macOS uses /Users/..., not /users/...
DROP_COLS = [
    "Unnamed: 0", "Unnamed.0", "property_id", "daft_id", "latitude", "longitude",
    "available_from", "address", "url", "input_date"
]
RENAME_MAP = {"distance_from_city_center": "distance"}

st.set_page_config(page_title="Interactive model plot (RF / XGBoost)", layout="wide")
st.title("Interactive model plot: Random Forest / XGBoost")

# --------------- Data loading ---------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    # Clean columns
    df = df.rename(columns=RENAME_MAP)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    # Convert obvious boolean strings to booleans
    for c in df.columns:
        if df[c].dtype == object:
            # If looks like date, leave as string (user can exclude)
            if "date" in c.lower():
                continue
            # Keep as object for now; encoders will handle it
    return df

with st.sidebar:
    st.header("Data & Options")
    data_path = st.text_input("CSV path", value=DEFAULT_DATA_PATH)
    mode = st.radio("Mode", options=("Single prediction", "What‑if sweep"), index=0)
    model_name = st.radio(
        "Model",
        options=("Random Forest",) + (("XGBoost",) if XGB_OK else tuple()),
        index=0,
        help=(None if XGB_OK else "Install xgboost to enable this option: pip install xgboost"),
    )
    npoints = st.slider("Points on plot (smoothness)", min_value=25, max_value=500, value=121, step=2)
    train_btn = st.button("Train / Re-train model")

# Load data
try:
    df_raw = load_data(data_path)
except Exception as e:
    st.error(str(e))
    st.stop()

st.write("**Data preview**")
st.dataframe(df_raw.head().astype(str), use_container_width=True)

# --------------- Target selection ---------------
# Require the user to choose a numeric target (no auto-selection)
num_candidates = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
left, right = st.columns([1, 2])
with left:
    target_choices = ["— pick a target —"] + num_candidates
    target_choice = st.selectbox("Target (numeric)", options=target_choices, index=0)
    if target_choice == "— pick a target —":
        st.info("Choose a numeric target to continue.")
        st.stop()
    target = target_choice

# --------------- Predictor selection ---------------
predictor_candidates = [c for c in df_raw.columns if c != target]
with left:
    use_preds = st.multiselect("Predictors to use", options=predictor_candidates, default=predictor_candidates)
    if not use_preds:
        st.warning("Select at least one predictor.")
        st.stop()

# Identify types
cat_cols = [c for c in use_preds if df_raw[c].dtype == object or isinstance(df_raw[c].dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(df_raw[c])]
num_cols_used = [c for c in use_preds if c not in cat_cols]

with left:
    vary_feat = st.selectbox("Feature to vary in plot", options=use_preds, index=0) if mode == "What‑if sweep" else None

# --------------- Widgets for fixed values ---------------
with right:
    if 'mode' in locals() and mode == "Single prediction":
        st.subheader("Set predictors")
        single_values = {}
        for c in use_preds:
            s = df_raw[c]
            if c in num_cols_used:
                vmin, vmax = float(np.nanmin(s)), float(np.nanmax(s))
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    continue
                median = float(np.nanmedian(s))
                step = max((vmax - vmin) / 100.0, 1e-6)
                single_values[c] = st.slider(c, min_value=vmin, max_value=vmax, value=median, step=step)
            else:
                choices = sorted([str(x) for x in pd.Series(s).dropna().unique()])
                if not choices:
                    continue
                default_level = str(pd.Series(s).astype(str).value_counts().idxmax())
                single_values[c] = st.selectbox(c, options=choices, index=choices.index(default_level) if default_level in choices else 0, key=f"sel_{c}")
    else:
        st.subheader("Set other predictors (held constant)")
        fixed_values = {}
        for c in use_preds:
            if c == vary_feat:
                continue
            s = df_raw[c]
            if c in num_cols_used:
                vmin, vmax = float(np.nanmin(s)), float(np.nanmax(s))
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    continue
                median = float(np.nanmedian(s))
                step = max((vmax - vmin) / 100.0, 1e-6)
                fixed_values[c] = st.slider(c, min_value=vmin, max_value=vmax, value=median, step=step)
            else:
                choices = sorted([str(x) for x in pd.Series(s).dropna().unique()])
                if not choices:
                    continue
                default_level = str(pd.Series(s).astype(str).value_counts().idxmax())
                fixed_values[c] = st.selectbox(c, options=choices, index=choices.index(default_level) if default_level in choices else 0, key=f"sel_{c}")

# --------------- Model choice ---------------
# Controls moved to sidebar: mode, model, train button

# --------------- Prepare training data ---------------
df = df_raw.dropna(subset=[target]).copy()
X = df[use_preds]
y = df[target]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), num_cols_used),
        ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ], remainder="drop"
)

if model_name == "Random Forest":
    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
else:
    # XGBoost sensible defaults
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
        verbosity=0,
    )

pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])

# Train on demand
if train_btn:
    with st.spinner("Training model…"):
        pipe.fit(X, y)
    st.success("Model trained.")
else:
    # Train once by default for convenience
    pipe.fit(X, y)

# --------------- Prediction(s) ---------------
if mode == "Single prediction":
    # Assemble a single row for prediction from controls
    one_row = {}
    for c in use_preds:
        if c in num_cols_used:
            one_row[c] = float(single_values[c])
        else:
            one_row[c] = str(single_values[c])
    one_df = pd.DataFrame([one_row])

    with st.spinner("Predicting…"):
        yhat_single = float(pipe.predict(one_df)[0])
        # Also predict on the whole training set to get a distribution
        yhat_all = pipe.predict(X)

    st.subheader("Prediction")
    st.metric(label=f"Predicted {target}", value=f"{yhat_single:,.2f}")

    # --- Distribution plots ---
    st.subheader("Prediction distribution")
    dfp = pd.DataFrame({"pred": yhat_all, "group": ["All"] * len(yhat_all)})
    tabs = st.tabs(["Boxplot", "Histogram", "Violin"])  # choose what you like

    # Boxplot with current prediction marked
    with tabs[0]:
        fig = px.box(dfp, x="group", y="pred", points=False, labels={"group": "", "pred": f"Predicted {target}"})
        # overlay current prediction as a diamond
        fig.add_scatter(x=["All"], y=[yhat_single], mode="markers", name="Current",
                        marker=dict(size=12, symbol="diamond-open"))
        st.plotly_chart(fig, use_container_width=True)

    # Histogram with vertical line at current prediction
    with tabs[1]:
        fig2 = px.histogram(dfp, x="pred", nbins=40, labels={"pred": f"Predicted {target}"})
        fig2.add_vline(x=yhat_single, line_dash="dash")
        st.plotly_chart(fig2, use_container_width=True)

    # Violin plot with current prediction marked
    with tabs[2]:
        fig3 = px.violin(dfp, x="group", y="pred", box=True, points=False,
                         labels={"group": "", "pred": f"Predicted {target}"})
        fig3.add_scatter(x=["All"], y=[yhat_single], mode="markers", name="Current",
                         marker=dict(size=12, symbol="diamond-open"))
        st.plotly_chart(fig3, use_container_width=True)

    # --- Optional: aggregated feature importance ---
    with st.expander("Feature importance (aggregated by original feature)"):
        try:
            prep = pipe.named_steps["prep"]
            mdl = pipe.named_steps["model"]
            importances = getattr(mdl, "feature_importances_", None)
            if importances is not None:
                # Get feature names from the preprocessor
                num_feats = num_cols_used
                cat_feats = cat_cols
                ohe = prep.named_transformers_["cat"].named_steps["onehot"] if cat_feats else None
                cat_names = ohe.get_feature_names_out(cat_feats).tolist() if ohe is not None else []
                feat_names = list(num_feats) + cat_names
                imp = pd.DataFrame({"feature": feat_names, "importance": importances})
                # Aggregate one-hot columns back to original categorical names
                def original_name(col):
                    if "__" in col:
                        # sklearn >=1.6 OneHotEncoder may use feature__level; older uses feature_level
                        return col.split("__", 1)[0]
                    if "_" in col and col.split("_", 1)[0] in set(cat_feats):
                        return col.split("_", 1)[0]
                    return col
                imp["orig"] = imp["feature"].apply(original_name)
                agg = imp.groupby("orig", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
                fig_imp = px.bar(agg, x="orig", y="importance", labels={"orig": "Feature", "importance": "Importance"})
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.caption("This model does not expose feature_importances_. Try Random Forest or XGBoost.")
        except Exception as e:
            st.caption(f"Could not compute feature importance: {e}")

else:
    # --- What-if sweep (previous behavior) ---
    def build_grid(df: pd.DataFrame, feature: str, npoints: int, fixed: dict) -> pd.DataFrame:
        grid = {}
        for c in use_preds:
            if c == feature:
                continue
            if c in fixed:
                grid[c] = [fixed[c]]
            else:
                s = df[c]
                if c in num_cols_used:
                    grid[c] = [float(np.nanmedian(s))]
                else:
                    grid[c] = [str(pd.Series(s).astype(str).value_counts().idxmax())]
        base = pd.DataFrame(grid)
        s = df[feature]
        if feature in num_cols_used:
            vmin, vmax = float(np.nanmin(s)), float(np.nanmax(s))
            xs = np.linspace(vmin, vmax, npoints)
            grid_df = pd.concat([base]*len(xs), ignore_index=True)
            grid_df[feature] = xs
        else:
            cats = sorted(pd.Series(s).dropna().astype(str).unique().tolist())
            if not cats:
                cats = [""]
            grid_df = pd.concat([base]*len(cats), ignore_index=True)
            grid_df[feature] = cats
        return grid_df

    pred_grid = build_grid(df, vary_feat, npoints, fixed_values)
    with st.spinner("Predicting…"):
        yhat = pipe.predict(pred_grid)
    pred_grid = pred_grid.assign(**{target: yhat})
    st.subheader("What‑if plot")
    if vary_feat in num_cols_used:
        fig = px.line(pred_grid, x=vary_feat, y=target, labels={vary_feat: vary_feat, target: f"Predicted {target}"})
    else:
        agg = pred_grid.groupby(vary_feat, as_index=False)[target].mean()
        fig = px.bar(agg, x=vary_feat, y=target, labels={vary_feat: vary_feat, target: f"Predicted {target}"})
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show current fixed predictor values"):
        if fixed_values:
            fixed_df = pd.DataFrame({
                "predictor": list(fixed_values.keys()),
                "value": list(fixed_values.values()),
            })
            fixed_df["value"] = fixed_df["value"].astype(str)
            st.dataframe(fixed_df, use_container_width=True)
        else:
            st.write("No fixed predictors set (only varying feature selected).")

# --------------- Hints ---------------
st.info(
    "Tips:\n"
    "• Increase 'Points on plot' for a smoother line.\n"
    "• Try switching model (RF vs XGBoost) to see different shapes.\n"
    "• Remove high-cardinality text columns from predictors if training is slow."
)
