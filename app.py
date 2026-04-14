"""
DataAnalytica: Multi-Agent Intelligent Data Analytics System
============================================================
Full-stack Streamlit ML analytics platform with:
  • Auto preprocessing pipeline (ColumnTransformer + Pipeline)
  • Expanded model selection (Classification + Regression)
  • Model performance diagnostics with explanations
  • Rich interactive Plotly dashboards
  • Dynamic, context-aware Q&A agent
  • 6-tab UI structure
  • Live agent logging panel
  • Modular, clean code structure
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json, time, copy, warnings
warnings.filterwarnings('ignore')

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                   LogisticRegression)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                              accuracy_score, classification_report, f1_score,
                              precision_score, recall_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ── Try XGBoost ───────────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DataAnalytica",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="◆"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;font-size:16px!important}
.stApp{background:#f7f5f2;color:#1a1a1a}
[data-testid="stSidebar"]{background:#1c1917!important;border-right:1px solid #292524}
[data-testid="stSidebar"] *{color:#a8a29e!important;font-size:15px!important}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#fafaf9!important;font-size:18px!important}
.block-container{background:#f7f5f2;padding:2rem 2.5rem;max-width:1400px}
h1{font-family:'Lora',serif!important;color:#1c1917!important;font-size:32px!important;font-weight:600!important}
h2{font-family:'Lora',serif!important;color:#292524!important;font-size:24px!important;font-weight:600!important}
h3{font-family:'Inter',sans-serif!important;color:#44403c!important;font-size:20px!important;font-weight:600!important}
p,li{font-size:16px!important;line-height:1.7!important}
[data-testid="metric-container"]{background:#ffffff!important;border:1px solid #e7e5e4!important;border-radius:10px!important;padding:18px 22px!important;box-shadow:0 1px 2px rgba(0,0,0,0.04)!important}
[data-testid="metric-container"] label{color:#78716c!important;font-size:13px!important;font-weight:600!important;letter-spacing:.6px!important;text-transform:uppercase!important}
[data-testid="stMetricValue"]{color:#1c1917!important;font-family:'JetBrains Mono',monospace!important;font-size:26px!important;font-weight:500!important}
[data-testid="stMetricDelta"]{color:#16a34a!important;font-size:14px!important}
[data-testid="stDataFrame"]{background:#ffffff!important;border:1px solid #e7e5e4!important;border-radius:10px!important}
[data-testid="stDataFrame"] *{font-size:15px!important}
.stButton>button{background:#1c1917!important;color:#fafaf9!important;border:none!important;border-radius:7px!important;font-family:'Inter',sans-serif!important;font-weight:500!important;font-size:15px!important;padding:10px 24px!important}
.stButton>button:hover{background:#292524!important}
[data-testid="stSelectbox"]>div>div,[data-testid="stTextInput"]>div>div>input{background:#ffffff!important;border:1px solid #d6d3d1!important;border-radius:7px!important;color:#1c1917!important;font-family:'Inter',sans-serif!important;font-size:15px!important}
[data-testid="stTabs"] [role="tab"]{background:transparent!important;color:#78716c!important;font-family:'Inter',sans-serif!important;font-weight:500!important;font-size:15px!important;border-radius:0!important;border-bottom:2px solid transparent!important;padding:10px 18px!important}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:#1c1917!important;border-bottom:2px solid #1c1917!important;font-weight:600!important}
[data-testid="stTabs"] [role="tablist"]{border-bottom:1px solid #e7e5e4!important}
[data-testid="stCheckbox"] label{font-size:15px!important;color:#a8a29e!important}
[data-testid="stMarkdownContainer"] p{font-size:15px!important}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-thumb{background:#d6d3d1;border-radius:3px}
</style>""", unsafe_allow_html=True)

# ── Plotly base layout ────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
    font=dict(color='#44403c', family='Inter,sans-serif', size=13),
    xaxis=dict(gridcolor='#f5f5f4', linecolor='#e7e5e4', tickcolor='#a8a29e',
               zeroline=False, tickfont=dict(size=12)),
    yaxis=dict(gridcolor='#f5f5f4', linecolor='#e7e5e4', tickcolor='#a8a29e',
               zeroline=False, tickfont=dict(size=12)),
    margin=dict(l=55, r=25, t=40, b=55),
)
C = dict(
    primary='#1c1917', orange='#ea580c', green='#16a34a', blue='#2563eb',
    teal='#0d9488', red='#dc2626', muted='#78716c', purple='#7c3aed',
    amber='#d97706', pink='#db2777'
)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
REG_MODELS: dict = {
    "Linear Regression":        LinearRegression(),
    "Ridge Regression":         Ridge(alpha=1.0),
    "Lasso Regression":         Lasso(alpha=0.1, max_iter=5000),
    "Decision Tree":            DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest":            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":        GradientBoostingRegressor(n_estimators=100, random_state=42),
    "K-Nearest Neighbors":      KNeighborsRegressor(n_neighbors=5),
    "Support Vector Regressor": SVR(kernel='rbf', C=1.0),
}
CLF_MODELS: dict = {
    "Logistic Regression":       LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":             DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":             RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":         GradientBoostingClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors":       KNeighborsClassifier(n_neighbors=5),
    "Support Vector Classifier": SVC(kernel='rbf', C=1.0, probability=True),
    "Naive Bayes":               GaussianNB(),
}
if _HAS_XGB:
    REG_MODELS["XGBoost Regressor"] = XGBRegressor(n_estimators=100, random_state=42,
                                                    verbosity=0, eval_metric='rmse')
    CLF_MODELS["XGBoost Classifier"] = XGBClassifier(n_estimators=100, random_state=42,
                                                      verbosity=0, eval_metric='logloss',
                                                      use_label_encoder=False)

MODEL_DESC = {
    "Linear Regression":         "Fits a straight line. Fast, interpretable, good baseline.",
    "Ridge Regression":          "Linear + L2 regularisation. Handles correlated features well.",
    "Lasso Regression":          "Linear + L1 regularisation. Can zero-out unimportant features.",
    "Decision Tree":             "Rule-based splits. Interpretable, captures non-linear patterns.",
    "Random Forest":             "Ensemble of many trees. Robust, strong all-rounder.",
    "Gradient Boosting":         "Sequential trees, each fixing the last. Top tabular accuracy.",
    "K-Nearest Neighbors":       "Predicts from the K most similar training samples.",
    "Support Vector Regressor":  "Margin-based fit. Good for smaller datasets.",
    "Logistic Regression":       "Probabilistic classifier. Fast, interpretable baseline.",
    "Support Vector Classifier": "Finds max-margin boundary. Robust on small data.",
    "Naive Bayes":               "Probabilistic model assuming feature independence. Very fast.",
    "XGBoost Regressor":         "Extreme gradient boosting — regularised, fast and highly accurate.",
    "XGBoost Classifier":        "XGBoost for classification with regularisation and speed.",
}

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════
_defaults = dict(
    df=None, analysis=None, chat_history=[], last_file=None,
    agent_status=None, ml_result=None, selected_models=[],
    selected_target=None, models_trained=False, task_type='regression',
    agent_log=[]
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if st.session_state.agent_status is None:
    st.session_state.agent_status = {
        a: 'idle' for a in ['intake', 'preprocess', 'eda', 'viz', 'ml', 'diagnostic', 'insight']
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data(uploaded_file) -> pd.DataFrame:
    """Load CSV from Streamlit uploader."""
    df = pd.read_csv(uploaded_file)
    return df


def infer_target(df: pd.DataFrame) -> str | None:
    keywords = ['target', 'label', 'price', 'score', 'result', 'output', 'sales',
                'revenue', 'amount', 'value', 'cost', 'rate', 'total', 'salary',
                'income', 'class', 'category', 'type', 'churn', 'fraud', 'survived']
    for kw in keywords:
        for col in df.columns:
            if kw in col.lower():
                return col
    ncols = df.select_dtypes(include='number').columns.tolist()
    if ncols:
        return ncols[-1]
    cats = df.select_dtypes(exclude='number').columns.tolist()
    return cats[-1] if cats else None


def detect_task(df: pd.DataFrame, target: str) -> str:
    if not target or target not in df.columns:
        return 'regression'
    col = df[target].dropna()
    if not pd.api.types.is_numeric_dtype(col):
        return 'classification'
    n_unique = col.nunique()
    if n_unique <= 20 and n_unique / max(len(col), 1) < 0.05:
        return 'classification'
    return 'regression'


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — AUTOMATIC PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def _add_log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.agent_log.append(f"[{ts}]  {msg}")


def build_classification_pipeline(X_df: pd.DataFrame):
    """
    ColumnTransformer pipeline for classification:
      numeric  → impute(mean)  → StandardScaler
      category → impute(mode)  → OneHotEncoder
    """
    num_cols = X_df.select_dtypes(include='number').columns.tolist()
    cat_cols = X_df.select_dtypes(exclude='number').columns.tolist()
    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler',  StandardScaler())
        ])
        transformers.append(('num', num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipe, cat_cols))
    return ColumnTransformer(transformers=transformers, remainder='drop')


def build_regression_pipeline(X_df: pd.DataFrame):
    """
    ColumnTransformer pipeline for regression:
      numeric  → impute(median) → StandardScaler
      category → impute(mode)  → LabelEncoder (via OHE + drop)
    """
    num_cols = X_df.select_dtypes(include='number').columns.tolist()
    cat_cols = X_df.select_dtypes(exclude='number').columns.tolist()
    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler())
        ])
        transformers.append(('num', num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipe, cat_cols))
    return ColumnTransformer(transformers=transformers, remainder='drop')


def preprocess_data(df: pd.DataFrame, target: str, task: str):
    """
    Full preprocessing: impute → encode → scale.
    Returns (X_array, y_array, preprocessor, feature_names, le_target or None).
    """
    _add_log("Preprocessing Agent: detecting numeric and categorical columns…")
    df2 = df.copy()

    # ── impute missing in target ──────────────────────────────────────────────
    if df2[target].isnull().any():
        if pd.api.types.is_numeric_dtype(df2[target]):
            df2[target].fillna(df2[target].median(), inplace=True)
        else:
            df2[target].fillna(df2[target].mode()[0], inplace=True)

    features = [c for c in df2.columns if c != target]
    X_df = df2[features]
    num_cols = X_df.select_dtypes(include='number').columns.tolist()
    cat_cols = X_df.select_dtypes(exclude='number').columns.tolist()

    _add_log(f"Preprocessing Agent: {len(num_cols)} numeric, {len(cat_cols)} categorical features detected.")

    if task == 'classification':
        preprocessor = build_classification_pipeline(X_df)
        le_target = LabelEncoder()
        y = le_target.fit_transform(df2[target].astype(str))
        X = preprocessor.fit_transform(X_df)
        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            feature_names = features
        _add_log("Preprocessing Agent: applied mean imputation + StandardScaler (numeric), mode imputation + OHE (categorical).")
        return X, y, preprocessor, feature_names, le_target, features
    else:
        preprocessor = build_regression_pipeline(X_df)
        y = df2[target].values.astype(float)
        X = preprocessor.fit_transform(X_df)
        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            feature_names = features
        _add_log("Preprocessing Agent: applied median imputation + StandardScaler (numeric), mode imputation + OHE (categorical).")
        return X, y, preprocessor, feature_names, None, features


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_models(df: pd.DataFrame, target: str, model_names: list) -> dict | None:
    """Train selected models; return results dict or None on failure."""
    task = detect_task(df, target)
    st.session_state.task_type = task
    _add_log(f"ML Agent: task auto-detected → {task.upper()}, target = '{target}'.")

    if len(df) < 20 or target not in df.columns:
        return None

    X, y, preprocessor, feature_names, le_target, raw_features = \
        preprocess_data(df, target, task)

    if task == 'classification':
        classes = le_target.classes_.tolist()
        n_classes = len(classes)
        stratify = y if n_classes > 1 and min(np.bincount(y)) > 1 else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify)
        avg = 'binary' if n_classes == 2 else 'weighted'
        trained = {}
        for name in model_names:
            model = CLF_MODELS.get(name)
            if model is None:
                continue
            _add_log(f"ML Agent: training {name}…")
            try:
                clf = copy.deepcopy(model)
                clf.fit(Xtr, ytr)
                preds = clf.predict(Xte)
                acc   = float(accuracy_score(yte, preds))
                f1    = float(f1_score(yte, preds, average=avg, zero_division=0))
                prec  = float(precision_score(yte, preds, average=avg, zero_division=0))
                rec   = float(recall_score(yte, preds, average=avg, zero_division=0))
                report_str = classification_report(yte, preds,
                                                   target_names=classes, zero_division=0)
                imp = _extract_importance(clf, feature_names)
                trained[name] = dict(
                    accuracy=acc, f1=f1, precision=prec, recall=rec,
                    report=report_str, importance=imp,
                    preds=le_target.inverse_transform(preds[:60]).tolist(),
                    y_test=le_target.inverse_transform(yte[:60]).tolist()
                )
                _add_log(f"ML Agent: {name} → Accuracy={acc:.4f}, F1={f1:.4f}")
            except Exception as e:
                trained[name] = dict(accuracy=0.0, f1=0.0, precision=0.0, recall=0.0,
                                     report='', importance=None, preds=[], y_test=[],
                                     error=str(e))
                _add_log(f"ML Agent: {name} failed — {e}")

        best_name = max(trained, key=lambda k: trained[k]['accuracy'])
        best = trained[best_name]
        return dict(
            task='classification', target=target, features=raw_features,
            feature_names=feature_names, models=trained,
            classes=classes, n_classes=n_classes,
            best=best_name, accuracy=best['accuracy'],
            f1=best['f1'], precision=best['precision'], recall=best['recall'],
            report=best['report'], n_train=len(Xtr), n_test=len(Xte),
            r2=None, mae=None, rmse=None,
            class_counts=pd.Series(y).value_counts().to_dict()
        )

    else:  # regression
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        trained = {}
        for name in model_names:
            model = REG_MODELS.get(name)
            if model is None:
                continue
            _add_log(f"ML Agent: training {name}…")
            try:
                clf = copy.deepcopy(model)
                clf.fit(Xtr, ytr)
                preds = clf.predict(Xte)
                imp = _extract_importance(clf, raw_features)
                trained[name] = dict(
                    r2=float(max(0.0, r2_score(yte, preds))),
                    mae=float(mean_absolute_error(yte, preds)),
                    rmse=float(np.sqrt(mean_squared_error(yte, preds))),
                    importance=imp,
                    preds=preds[:60].tolist(),
                    y_test=yte[:60].tolist()
                )
                _add_log(f"ML Agent: {name} → R²={trained[name]['r2']:.4f}, RMSE={trained[name]['rmse']:.4f}")
            except Exception as e:
                trained[name] = dict(r2=0.0, mae=0.0, rmse=0.0, importance=None,
                                     preds=[], y_test=[], error=str(e))
                _add_log(f"ML Agent: {name} failed — {e}")

        best_name = max(trained, key=lambda k: trained[k]['r2'])
        best = trained[best_name]
        return dict(
            task='regression', target=target, features=raw_features,
            feature_names=feature_names, models=trained,
            best=best_name, r2=best['r2'], mae=best['mae'], rmse=best['rmse'],
            n_train=len(Xtr), n_test=len(Xte),
            accuracy=None, f1=None, precision=None, recall=None
        )


def _extract_importance(clf, feature_names: list) -> dict | None:
    try:
        if hasattr(clf, 'feature_importances_'):
            return dict(zip(feature_names, clf.feature_importances_.tolist()))
        if hasattr(clf, 'coef_'):
            coef = clf.coef_
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            return dict(zip(feature_names, np.abs(coef).tolist()))
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — INITIAL PIPELINE (EDA)
# ══════════════════════════════════════════════════════════════════════════════
def run_initial_pipeline(df: pd.DataFrame) -> dict:
    result = {}
    _add_log("Intake Agent: reading dataset schema, types and completeness…")
    st.session_state.agent_status['intake'] = 'running'
    missing = int(df.isnull().sum().sum())
    total   = df.shape[0] * df.shape[1]
    result['intake'] = dict(
        rows=df.shape[0], cols=df.shape[1],
        numeric=df.select_dtypes(include='number').columns.tolist(),
        categorical=df.select_dtypes(exclude='number').columns.tolist(),
        missing=missing,
        completeness=round((1 - missing / total) * 100, 1) if total > 0 else 100.0
    )
    _add_log(f"Intake Agent: {result['intake']['rows']:,} rows, {result['intake']['cols']} cols, "
             f"{result['intake']['completeness']}% complete.")
    st.session_state.agent_status['intake'] = 'done'

    st.session_state.agent_status['preprocess'] = 'running'
    _add_log("Preprocessing Agent: computing missing-value map and dtype summary…")
    result['missing_by_col'] = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
    st.session_state.agent_status['preprocess'] = 'done'

    st.session_state.agent_status['eda'] = 'running'
    _add_log("EDA Agent: computing descriptive statistics and correlation matrix…")
    ncols = result['intake']['numeric']
    result['eda'] = dict(
        stats=df[ncols].describe() if ncols else None,
        corr=df[ncols].corr() if len(ncols) > 1 else None,
        ncols=ncols
    )
    st.session_state.agent_status['eda'] = 'done'

    st.session_state.agent_status['viz'] = 'running'
    _add_log("Visualization Agent: building chart data for heatmap, distributions, scatter…")
    time.sleep(0.05)
    st.session_state.agent_status['viz'] = 'done'

    result['ml'] = None
    result['insights'] = []
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
def generate_visualizations(df: pd.DataFrame, eda: dict, target: str | None):
    """Render all visualization tabs inline (called from Tab 2)."""
    ncols = eda['ncols']
    if not ncols:
        st.info("No numeric columns found for visualization.")
        return

    # ── Section: Data Exploration ─────────────────────────────────────────────
    st.markdown("## 🔍 Data Exploration")
    c_a, c_b = st.columns(2)

    with c_a:
        st.markdown("### Correlation Heatmap")
        st.markdown("<p style='color:#78716c;font-size:14px;margin-top:-6px;margin-bottom:10px'>Strength of linear relationships between numeric features. Deep blue = strong positive; deep red = strong negative.</p>", unsafe_allow_html=True)
        if eda['corr'] is not None and len(ncols) > 1:
            corr_df = eda['corr'].round(2)
            labels  = [c[:13] for c in corr_df.columns]
            n = len(labels)
            txt_sz = max(8, min(12, int(100 / max(n, 1))))
            fig_h = go.Figure(go.Heatmap(
                z=corr_df.values, x=labels, y=labels,
                colorscale=[[0, '#dc2626'], [0.5, '#f7f5f2'], [1, '#1d4ed8']],
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr_df.values],
                texttemplate="%{text}", textfont={"size": txt_sz, "color": "#44403c"},
                colorbar=dict(thickness=12, tickfont=dict(size=11, color='#78716c'))
            ))
            fig_h.update_layout(**PL, height=360)
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Need ≥ 2 numeric columns for correlation heatmap.")

    with c_b:
        st.markdown("### Missing Value Map")
        st.markdown("<p style='color:#78716c;font-size:14px;margin-top:-6px;margin-bottom:10px'>Percentage of missing values per column. Columns with 0% are omitted.</p>", unsafe_allow_html=True)
        miss_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
        miss_pct = miss_pct[miss_pct > 0]
        if len(miss_pct) == 0:
            st.success("✅ No missing values in this dataset!")
        else:
            fig_mv = go.Figure(go.Bar(
                x=miss_pct.index.tolist(),
                y=miss_pct.values.tolist(),
                marker_color=C['red'], opacity=0.8,
                text=[f"{v:.1f}%" for v in miss_pct.values],
                textposition='outside', textfont=dict(size=12, color='#44403c')
            ))
            fig_mv.update_layout(**PL, height=320, yaxis_title="Missing %", showlegend=False)
            st.plotly_chart(fig_mv, use_container_width=True)

    # Distribution + Boxplots
    c_c, c_d = st.columns(2)
    with c_c:
        st.markdown("### Feature Distribution")
        sel_dist = st.selectbox("Select column", ncols, key="dist_col")
        col_data = df[sel_dist].dropna()
        skew_val = float(col_data.skew())
        skew_note = ("symmetric" if abs(skew_val) < 0.5
                     else "moderately skewed" if abs(skew_val) < 1.0 else "heavily skewed")
        st.markdown(f"<p style='color:#78716c;font-size:14px;margin-top:-6px;margin-bottom:10px'>"
                    f"<strong>{sel_dist}</strong> — {skew_note} (skewness={skew_val:.2f}).</p>",
                    unsafe_allow_html=True)
        q1 = float(col_data.quantile(0.25)); q3 = float(col_data.quantile(0.75)); iqr = q3 - q1
        bw = 2 * iqr / (len(col_data) ** (1/3)) if iqr > 0 else 1
        n_bins = max(10, min(60, int((col_data.max() - col_data.min()) / bw) if bw > 0 else 25))
        fig_d = go.Figure(go.Histogram(
            x=col_data, nbinsx=n_bins,
            marker_color=C['orange'], opacity=0.75,
            marker_line_color='#c2410c', marker_line_width=0.4
        ))
        fig_d.update_layout(**PL, height=300, xaxis_title=sel_dist,
                            yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig_d, use_container_width=True)

    with c_d:
        st.markdown("### Spread & Outliers (Boxplot)")
        box_sel = st.multiselect("Columns", ncols, default=ncols[:5], key="box_sel")
        if box_sel:
            box_clrs = [C['primary'], C['orange'], C['green'], C['blue'],
                        C['purple'], C['red'], C['teal'], C['amber']]
            fig_box = go.Figure()
            for i, col in enumerate(box_sel):
                clr = box_clrs[i % len(box_clrs)]
                fig_box.add_trace(go.Box(
                    y=df[col].dropna(), name=col[:14],
                    marker_color=clr, line_color=clr,
                    fillcolor='rgba(255,255,255,0.9)',
                    boxpoints='outliers', jitter=0.3
                ))
            fig_box.update_layout(**PL, height=300, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Select at least one column.")

    # ── Section: Feature Relationships ────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔗 Feature Relationships")

    c_e, c_f = st.columns(2)
    with c_e:
        st.markdown("### Scatter — Feature vs Target")
        cur_target = target
        feat_opts = [c for c in ncols if c != cur_target] if cur_target else ncols
        if feat_opts and cur_target and cur_target in df.columns:
            x_sel = st.selectbox("X-axis feature", feat_opts, key="scatter_x")
            try:
                corr_val = float(df[[x_sel, cur_target]].corr().iloc[0, 1])
            except Exception:
                corr_val = 0.0
            direction = "positive" if corr_val > 0.1 else "negative" if corr_val < -0.1 else "little"
            st.markdown(f"<p style='color:#78716c;font-size:14px;margin-top:-6px;margin-bottom:10px'>"
                        f"<strong>{x_sel}</strong> ↔ <strong>{cur_target}</strong>: r={corr_val:.3f} — {direction} association.</p>",
                        unsafe_allow_html=True)
            fig_s = go.Figure(go.Scatter(
                x=df[x_sel], y=df[cur_target], mode='markers',
                opacity=0.45, marker=dict(color=C['blue'], size=5)
            ))
            fig_s.update_layout(**PL, height=300, xaxis_title=x_sel,
                                yaxis_title=cur_target, showlegend=False)
            st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.info("Select a valid target column in the sidebar.")

    with c_f:
        st.markdown("### Feature Scale Comparison")
        st.markdown("<p style='color:#78716c;font-size:14px;margin-top:-6px;margin-bottom:10px'>Mean, Min and Max for each numeric feature. Models using distance metrics benefit from standardisation.</p>", unsafe_allow_html=True)
        disp_cols = ncols[:8]
        stats_df2 = df[disp_cols].describe()
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(name='Mean', x=[c[:13] for c in disp_cols],
                               y=stats_df2.loc['mean'].round(3), marker_color=C['primary'], opacity=0.85))
        fig_b.add_trace(go.Bar(name='Min',  x=[c[:13] for c in disp_cols],
                               y=stats_df2.loc['min'].round(3),  marker_color=C['red'],    opacity=0.6))
        fig_b.add_trace(go.Bar(name='Max',  x=[c[:13] for c in disp_cols],
                               y=stats_df2.loc['max'].round(3),  marker_color=C['green'],  opacity=0.6))
        fig_b.update_layout(**PL, barmode='group', height=300,
                            legend=dict(font=dict(color='#44403c', size=13),
                                        bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_b, use_container_width=True)

    # ── Pair plot (numeric sample) ─────────────────────────────────────────────
    if len(ncols) >= 2:
        st.markdown("### Pairwise Scatter Matrix")
        pair_cols = st.multiselect("Select features (2–5 recommended)",
                                   ncols, default=ncols[:min(4, len(ncols))], key="pair_cols")
        if len(pair_cols) >= 2:
            samp = df[pair_cols].dropna().sample(min(500, len(df)), random_state=42)
            fig_pair = px.scatter_matrix(
                samp, dimensions=pair_cols,
                color_discrete_sequence=[C['blue']]
            )
            fig_pair.update_traces(marker=dict(size=3, opacity=0.35))
            fig_pair.update_layout(
                paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                font=dict(color='#44403c', family='Inter', size=11),
                height=460, margin=dict(l=40, r=20, t=40, b=40)
            )
            st.plotly_chart(fig_pair, use_container_width=True)
        else:
            st.info("Select at least 2 columns for pair plot.")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — MODEL DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
def generate_diagnostics(df: pd.DataFrame, ml: dict | None) -> list[dict]:
    """
    Analyse WHY performance may be low and return structured diagnostic items.
    """
    _add_log("Diagnostic Agent: running model performance diagnostics…")
    diags = []
    n_rows, n_cols = df.shape
    missing_pct = df.isnull().mean().mean() * 100

    # ── Sample size ───────────────────────────────────────────────────────────
    if n_rows < 500:
        diags.append({
            "icon": "⚠️", "severity": "warning", "title": "Small dataset",
            "body": (f"The dataset has only {n_rows} rows. Complex models like Gradient Boosting "
                     f"and Neural Networks may overfit. Prefer simpler models (Logistic Regression, "
                     f"Ridge) and use cross-validation instead of a single train/test split."),
            "suggestion": "Use Random Forest or Ridge Regression for better generalisation."
        })
    elif n_rows < 2000:
        diags.append({
            "icon": "ℹ️", "severity": "info", "title": "Moderate dataset size",
            "body": (f"Dataset has {n_rows} rows — adequate for most models but ensemble methods "
                     f"may benefit from more data. Consider data augmentation or feature engineering."),
            "suggestion": "Gradient Boosting + cross-validation is a reliable choice here."
        })
    else:
        diags.append({
            "icon": "✅", "severity": "success", "title": "Adequate dataset size",
            "body": (f"With {n_rows:,} samples, you have sufficient data to reliably train "
                     f"complex models. Ensemble methods should work well."),
            "suggestion": "Consider XGBoost or Gradient Boosting for best accuracy."
        })

    # ── Missing values ────────────────────────────────────────────────────────
    if missing_pct > 20:
        diags.append({
            "icon": "🔴", "severity": "warning", "title": "High proportion of missing values",
            "body": (f"{missing_pct:.1f}% of cells are missing. The automatic preprocessing pipeline "
                     f"imputes numeric columns with the median and categorical columns with the mode, "
                     f"but this may introduce bias if data is not missing at random (MNAR)."),
            "suggestion": "Investigate WHY values are missing — consider dropping columns with >40% missing."
        })
    elif missing_pct > 0:
        diags.append({
            "icon": "⚠️", "severity": "info", "title": "Some missing values detected",
            "body": (f"{missing_pct:.1f}% missing values automatically imputed during preprocessing."),
            "suggestion": "Imputation is handled automatically but verify imputed values look sensible."
        })
    else:
        diags.append({
            "icon": "✅", "severity": "success", "title": "No missing values",
            "body": "All cells are filled — no imputation was needed.",
            "suggestion": ""
        })

    # ── ML-specific diagnostics ───────────────────────────────────────────────
    if ml:
        task = ml.get('task', 'regression')

        if task == 'classification':
            acc = ml['accuracy']

            # Class imbalance
            class_counts = ml.get('class_counts', {})
            if class_counts:
                counts = list(class_counts.values())
                imbalance_ratio = max(counts) / max(min(counts), 1)
                if imbalance_ratio > 3:
                    diags.append({
                        "icon": "⚖️", "severity": "warning", "title": "Class imbalance detected",
                        "body": (f"The most frequent class has {imbalance_ratio:.1f}x more samples "
                                 f"than the least frequent. This can cause the model to be biased "
                                 f"toward the majority class, inflating accuracy but hurting F1/Recall."),
                        "suggestion": "Try class_weight='balanced', SMOTE oversampling, or use F1 score as primary metric."
                    })

            # Low accuracy
            if acc < 0.60:
                diags.append({
                    "icon": "🔴", "severity": "warning", "title": "Low classification accuracy",
                    "body": (f"Best model accuracy is {acc:.4f} ({acc*100:.1f}%). Possible causes: "
                             f"insufficient discriminative features, noisy labels, severe class imbalance, "
                             f"or underfitting. The number of features ({len(ml['features'])}) relative "
                             f"to samples ({ml['n_train']}) may also be causing issues."),
                    "suggestion": "Try feature engineering, remove noisy features, or use Gradient Boosting / XGBoost."
                })
            elif acc < 0.80:
                diags.append({
                    "icon": "⚠️", "severity": "info", "title": "Moderate classification accuracy",
                    "body": (f"Accuracy of {acc:.4f} is moderate. There is room to improve with "
                             f"hyperparameter tuning, more features, or ensemble stacking."),
                    "suggestion": "Grid search on Random Forest or Gradient Boosting may push accuracy higher."
                })
            else:
                diags.append({
                    "icon": "✅", "severity": "success", "title": "Strong classification accuracy",
                    "body": (f"Accuracy of {acc:.4f} ({acc*100:.1f}%) is strong. The model reliably "
                             f"classifies '{ml['target']}'. Verify performance on an independent test set."),
                    "suggestion": "Validate with k-fold cross-validation to confirm robustness."
                })

            # Overfitting check (proxy: very few test samples)
            if ml['n_test'] < 30:
                diags.append({
                    "icon": "⚠️", "severity": "warning", "title": "Very small test set",
                    "body": (f"Only {ml['n_test']} samples in the test set — results may be unreliable. "
                             f"With small datasets, a single accuracy number has high variance."),
                    "suggestion": "Use 5-fold or 10-fold cross-validation for more reliable estimates."
                })

        else:  # regression
            r2 = ml['r2']
            if r2 < 0.3:
                diags.append({
                    "icon": "🔴", "severity": "warning", "title": "Weak regression performance",
                    "body": (f"Best R²={r2:.4f} — the model explains only {r2*100:.1f}% of variance in "
                             f"'{ml['target']}'. Possible causes: non-linear relationships not captured, "
                             f"irrelevant features, high noise in the target, or too few informative features."),
                    "suggestion": "Try Gradient Boosting or XGBoost. Consider polynomial feature engineering or log-transforming the target."
                })
            elif r2 < 0.6:
                diags.append({
                    "icon": "⚠️", "severity": "info", "title": "Moderate regression performance",
                    "body": (f"R²={r2:.4f} explains {r2*100:.1f}% of variance. Real signal exists "
                             f"but the model misses some patterns. Non-linear models or additional features "
                             f"could help."),
                    "suggestion": "Try Gradient Boosting with hyperparameter tuning. Check residual plots for patterns."
                })
            else:
                diags.append({
                    "icon": "✅", "severity": "success", "title": "Strong regression performance",
                    "body": (f"R²={r2:.4f} — the model explains {r2*100:.1f}% of variance in "
                             f"'{ml['target']}'. This is a strong result for tabular data."),
                    "suggestion": "Run cross-validation to confirm the score is stable across folds."
                })

    # ── Feature-to-sample ratio ───────────────────────────────────────────────
    if ml and len(ml['features']) > n_rows / 10:
        diags.append({
            "icon": "⚠️", "severity": "warning", "title": "High feature-to-sample ratio",
            "body": (f"{len(ml['features'])} features vs {n_rows} rows — ratio is "
                     f"{len(ml['features'])/max(n_rows,1):.2f}. High-dimensional data relative "
                     f"to samples can cause overfitting, especially for linear models."),
            "suggestion": "Apply PCA, remove low-importance features, or use Lasso/Ridge regularisation."
        })

    _add_log(f"Diagnostic Agent: generated {len(diags)} diagnostic findings.")
    return diags


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def get_insights(df: pd.DataFrame, ml: dict | None) -> list[dict]:
    _add_log("Insight Agent: generating data insights…")
    try:
        from groq import Groq
        if GROQ_API_KEY.strip() == "YOUR_GROQ_API_KEY_HERE":
            raise ValueError("no key")
        client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        return _fallback_insights(df, ml)

    ncols  = df.select_dtypes(include='number').columns.tolist()
    stats_s = df[ncols[:5]].describe().round(3).to_string() if ncols else "No numeric cols"
    if ml:
        t = ml.get('task', 'regression')
        ml_s = (f"Task:Classification,Target:{ml['target']},Best:{ml['best']},"
                f"Accuracy:{ml['accuracy']:.3f},F1:{ml['f1']:.3f},Classes:{ml.get('classes')}"
                if t == 'classification' else
                f"Task:Regression,Target:{ml['target']},Best:{ml['best']},"
                f"R²:{ml['r2']:.3f},RMSE:{ml['rmse']:.3f},MAE:{ml['mae']:.3f}")
    else:
        ml_s = "No ML run yet"

    prompt = (f"You are an expert data analyst. Write 4 specific, data-driven insights.\n"
              f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols.\n"
              f"Stats:\n{stats_s}\nML: {ml_s}\n"
              f"Rules: Reference actual column names and exact numbers. Be specific. "
              f"Each insight must be different — cover data quality, patterns, ML performance, and recommendations.\n"
              f'Return ONLY valid JSON array: [{{"title":"...","body":"2-3 sentences.","type":"info|success|warning|trend"}}]')
    try:
        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900, temperature=0.35
        )
        raw = resp.choices[0].message.content.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(raw)
    except Exception:
        return _fallback_insights(df, ml)


def _fallback_insights(df: pd.DataFrame, ml: dict | None) -> list[dict]:
    ncols = df.select_dtypes(include='number').columns.tolist()
    completeness = round((1 - df.isnull().mean().mean()) * 100, 1)
    out = [
        {"title": "Data completeness",
         "body": (f"The dataset has {df.shape[0]:,} records and {df.shape[1]} columns, "
                  f"with {completeness}% of values present. "
                  + ("No missing values — data is ready to use."
                     if df.isnull().sum().sum() == 0
                     else f"{df.isnull().sum().sum()} missing values were auto-imputed during preprocessing.")),
         "type": "success" if completeness > 95 else "warning"},
        {"title": "Feature composition",
         "body": (f"There are {len(ncols)} numeric and {df.shape[1]-len(ncols)} categorical columns. "
                  "Numeric features were standardised; categorical features were one-hot encoded. "
                  "All preprocessing was handled automatically by the pipeline."),
         "type": "info"},
    ]
    if ml:
        t = ml.get('task', 'regression')
        if t == 'classification':
            acc = ml['accuracy']
            quality = "strong" if acc > 0.85 else "moderate" if acc > 0.65 else "needs improvement"
            out.append({"title": "Classification performance",
                        "body": (f"Best model ({ml['best']}) — Accuracy={acc:.3f}, F1={ml['f1']:.3f}. "
                                 f"This is {quality}. Classes: {', '.join(str(c) for c in ml.get('classes', [])[:6])}."),
                        "type": "success" if acc > 0.85 else "info" if acc > 0.65 else "warning"})
        else:
            r2 = ml['r2']
            quality = "strong" if r2 > 0.7 else "moderate" if r2 > 0.4 else "weak"
            out.append({"title": "Regression performance",
                        "body": (f"Best model ({ml['best']}) achieves R²={r2:.3f} on '{ml['target']}', "
                                 f"explaining {r2*100:.1f}% of variance. This is {quality}."),
                        "type": "success" if r2 > 0.7 else "info" if r2 > 0.4 else "warning"})
    out.append({"title": "Recommended next steps",
                "body": ("Examine the feature importance chart to identify the most influential variables. "
                         "Consider running cross-validation for more reliable performance estimates. "
                         "Check the Diagnostics tab for specific improvement recommendations."),
                "type": "trend"})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 8 — QUESTION ANSWERING AGENT
# ══════════════════════════════════════════════════════════════════════════════
def build_context(df: pd.DataFrame, eda: dict, intake: dict,
                  ml: dict | None, task: str) -> str:
    ncols_list = eda['ncols']
    stats_str  = df[ncols_list[:6]].describe().round(3).to_string() if ncols_list else "None"
    corr_str   = ""
    if eda['corr'] is not None:
        corr = eda['corr']
        cc = corr.columns.tolist()
        pairs = []
        for i in range(len(cc)):
            for j in range(i + 1, len(cc)):
                pairs.append((abs(corr.iloc[i, j]), cc[i], cc[j], corr.iloc[i, j]))
        pairs.sort(reverse=True)
        corr_str = "\nTopCorrelations: " + "; ".join(
            f"{a}↔{b}={v:.3f}" for _, a, b, v in pairs[:6])

    if ml:
        t = ml.get('task', 'regression')
        if t == 'classification':
            all_m = "; ".join(
                f"{n}:Acc={v['accuracy']:.4f},F1={v['f1']:.4f}" for n, v in ml['models'].items())
            br = ml['models'][ml['best']]
            imp_str = ""
            if br.get('importance'):
                top5 = sorted(br['importance'].items(), key=lambda x: x[1], reverse=True)[:5]
                imp_str = "TopFeatures: " + ", ".join(f"{n}={v:.4f}" for n, v in top5)
            cc_str = str(ml.get('class_counts', {}))
            ml_ctx = (f"Task=Classification,Target={ml['target']},Best={ml['best']},"
                      f"Accuracy={ml['accuracy']:.4f},F1={ml['f1']:.4f},"
                      f"Precision={ml['precision']:.4f},Recall={ml['recall']:.4f},"
                      f"Classes={ml.get('classes')},ClassCounts={cc_str},"
                      f"Train={ml['n_train']},Test={ml['n_test']},{imp_str}\nAllModels:{all_m}")
        else:
            all_m = "; ".join(
                f"{n}:R²={v['r2']:.4f},MAE={v['mae']:.4f},RMSE={v['rmse']:.4f}"
                for n, v in ml['models'].items())
            br = ml['models'][ml['best']]
            imp_str = ""
            if br.get('importance'):
                top5 = sorted(br['importance'].items(), key=lambda x: x[1], reverse=True)[:5]
                imp_str = "TopFeatures: " + ", ".join(f"{n}={v:.4f}" for n, v in top5)
            ml_ctx = (f"Task=Regression,Target={ml['target']},Best={ml['best']},"
                      f"R²={ml['r2']:.4f},RMSE={ml['rmse']:.4f},MAE={ml['mae']:.4f},"
                      f"Train={ml['n_train']},Test={ml['n_test']},Features={ml['features']},"
                      f"{imp_str}\nAllModels:{all_m}")
    else:
        ml_ctx = "No ML run yet"

    return (f"Dataset:{intake['rows']}rows x {intake['cols']}cols.\n"
            f"Numeric:{', '.join(intake['numeric'][:10])}.\n"
            f"Categorical:{', '.join(intake['categorical'][:8])}.\n"
            f"Completeness:{intake['completeness']}%,Missing:{intake['missing']}.\n"
            f"Stats:\n{stats_str}{corr_str}\nML:{ml_ctx}")


def ask_ai_question(question: str, context: str, df: pd.DataFrame,
                    eda: dict, intake: dict, ml: dict | None, task: str) -> str:
    """
    Dynamic Q&A: tries Groq LLM first with full dataset context,
    falls back to rich rule-based answering engine.
    """
    prompt = (f"You are a senior data scientist explaining results to a student.\n"
              f"Answer in plain English. Always quote exact numbers from the context.\n"
              f"3-5 clear sentences, no bullet points.\n\n"
              f"=== DATASET CONTEXT ===\n{context}\n\n"
              f"=== QUESTION ===\n{question}")

    groq_key = GROQ_API_KEY.strip()
    if groq_key and groq_key != "YOUR_GROQ_API_KEY_HERE":
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600, temperature=0.4
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    return _local_answer(question, df, eda, intake, ml, task)


def _local_answer(question: str, df: pd.DataFrame, eda: dict,
                  intake: dict, ml: dict | None, task: str) -> str:
    q = question.lower()

    # ── Which feature has highest correlation with target? ─────────────────
    if any(x in q for x in ['correlation with target', 'corr with target',
                             'most correlated', 'highest correlation', 'most related']):
        tgt = ml['target'] if ml else None
        if tgt and eda['corr'] is not None and tgt in eda['corr'].columns:
            corr_col = eda['corr'][tgt].drop(tgt).abs().sort_values(ascending=False)
            top_feat = corr_col.index[0]
            top_val  = eda['corr'][tgt][top_feat]
            return (f"The feature with the highest correlation with '{tgt}' is "
                    f"'{top_feat}' (r={top_val:.4f}). "
                    f"{'A positive correlation means they increase together.' if top_val > 0 else 'A negative correlation means as one increases, the other decreases.'} "
                    f"Other highly correlated features: "
                    f"{', '.join(f'{c}={eda[chr(99)+(chr(111))+(chr(114))+(chr(114))][tgt][c]:.3f}' for c in corr_col.index[1:4]) if len(corr_col) > 1 else 'none'}.")
        elif eda['corr'] is not None:
            cc = eda['corr']; cols = cc.columns.tolist()
            pairs = []
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pairs.append((abs(cc.iloc[i, j]), cols[i], cols[j], cc.iloc[i, j]))
            pairs.sort(reverse=True)
            a, b, v = pairs[0][1], pairs[0][2], pairs[0][3]
            return f"The strongest pair-wise correlation is between '{a}' and '{b}' (r={v:.4f})."
        return "No numeric correlation data available."

    # ── Is the dataset balanced? ────────────────────────────────────────────
    if any(x in q for x in ['balanced', 'imbalanced', 'class balance', 'class distribution']):
        if ml and ml.get('task') == 'classification':
            cc = ml.get('class_counts', {})
            if cc:
                vals = list(cc.values())
                ratio = max(vals) / max(min(vals), 1)
                balance = "well balanced" if ratio < 1.5 else "moderately imbalanced" if ratio < 3 else "highly imbalanced"
                dist = ", ".join(f"'{k}': {v}" for k, v in cc.items())
                return (f"The target '{ml['target']}' is {balance} (imbalance ratio={ratio:.1f}x). "
                        f"Class distribution: {dist}. "
                        f"{'No special handling is needed.' if ratio < 1.5 else 'Consider class_weight=balanced or SMOTE oversampling to handle this imbalance.'}")
        return "Upload and run a classification task to see class balance information."

    # ── Which model performed best? ─────────────────────────────────────────
    if any(x in q for x in ['best model', 'which model', 'model performed', 'top model']):
        if ml:
            if task == 'classification':
                ranking = sorted(ml['models'].items(), key=lambda x: x[1]['accuracy'], reverse=True)
                scores = "; ".join(f"{n}: Acc={v['accuracy']:.4f}" for n, v in ranking)
                return (f"The best model is {ml['best']} with Accuracy={ml['accuracy']:.4f} "
                        f"and F1={ml['f1']:.4f}. All model rankings: {scores}.")
            else:
                ranking = sorted(ml['models'].items(), key=lambda x: x[1]['r2'], reverse=True)
                scores = "; ".join(f"{n}: R²={v['r2']:.4f}" for n, v in ranking)
                return (f"The best model is {ml['best']} with R²={ml['r2']:.4f} "
                        f"and RMSE={ml['rmse']:.4f}. All model rankings: {scores}.")
        return "No models have been trained yet. Select models in the sidebar and click Run Models."

    # ── Most important variables? ────────────────────────────────────────────
    if any(x in q for x in ['important', 'feature importance', 'most influential',
                             'key variable', 'which feature', 'top feature']):
        if ml:
            br = ml['models'][ml['best']]
            if br.get('importance'):
                top = sorted(br['importance'].items(), key=lambda x: x[1], reverse=True)[:5]
                return (f"Top 5 most important features for {ml['best']} predicting '{ml['target']}': "
                        + ", ".join(f"'{n}' (score={v:.4f})" for n, v in top)
                        + ". Higher score = more influence on predictions.")
            return f"{ml['best']} does not expose feature importances (e.g., SVM). Use Random Forest or Gradient Boosting to see importance scores."
        return "Train a model first to see feature importances."

    # ── What patterns exist in the data? ────────────────────────────────────
    if any(x in q for x in ['pattern', 'trend', 'insight', 'finding', 'interesting']):
        lines = [f"The dataset has {intake['rows']:,} rows and {intake['cols']} columns."]
        if eda['corr'] is not None:
            cc = eda['corr']; cols = cc.columns.tolist()
            pairs = []
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pairs.append((abs(cc.iloc[i, j]), cols[i], cols[j], cc.iloc[i, j]))
            pairs.sort(reverse=True)
            if pairs:
                a, b, v = pairs[0][1], pairs[0][2], pairs[0][3]
                lines.append(f"Strongest relationship: '{a}' and '{b}' (r={v:.3f}).")
        if intake['missing'] > 0:
            lines.append(f"{intake['missing']} missing values were detected and auto-imputed.")
        if ml:
            lines.append(f"The best ML model ({ml['best']}) achieves "
                         + (f"Accuracy={ml['accuracy']:.4f}" if task == 'classification'
                            else f"R²={ml['r2']:.4f}") + ".")
        return " ".join(lines)

    # ── Classification metrics ────────────────────────────────────────────────
    if task == 'classification' and ml:
        if any(x in q for x in ['accuracy', 'acc', 'correct', 'score']):
            ms = "; ".join(f"{n}: {v['accuracy']:.4f}" for n, v in ml['models'].items())
            q_str = "strong (>85%)" if ml['accuracy'] > 0.85 else "moderate (65–85%)" if ml['accuracy'] > 0.65 else "low (<65%)"
            return (f"Best model: {ml['best']} — Accuracy={ml['accuracy']:.4f} "
                    f"({ml['accuracy']*100:.1f}% correct on the test set for '{ml['target']}'). "
                    f"This is {q_str}. All models: {ms}.")
        if any(x in q for x in ['f1', 'precision', 'recall']):
            return (f"{ml['best']}: F1={ml['f1']:.4f}, Precision={ml['precision']:.4f}, Recall={ml['recall']:.4f}. "
                    "Precision = fraction of predicted positives that are truly positive. "
                    "Recall = fraction of actual positives the model found. "
                    "F1 is the harmonic mean — ideal when classes are imbalanced.")

    # ── Regression metrics ────────────────────────────────────────────────────
    if task == 'regression' and ml:
        if any(x in q for x in ['r2', 'r²', 'r-squared', 'variance', 'explain']):
            ms = "; ".join(f"{n}: R²={v['r2']:.4f}" for n, v in ml['models'].items())
            q_str = "strong (>0.7)" if ml['r2'] > 0.7 else "moderate (0.4–0.7)" if ml['r2'] > 0.4 else "weak (<0.4)"
            return (f"Best model: {ml['best']} — R²={ml['r2']:.4f}, explaining "
                    f"{ml['r2']*100:.1f}% of variance in '{ml['target']}'. "
                    f"This is {q_str}. All models: {ms}.")
        if any(x in q for x in ['rmse', 'mae', 'error']):
            return (f"{ml['best']}: MAE={ml['mae']:.4f}, RMSE={ml['rmse']:.4f} "
                    f"(both in units of '{ml['target']}'). MAE = average absolute error. "
                    "RMSE penalises larger errors more heavily. Lower = better predictions.")

    # ── General definitions ───────────────────────────────────────────────────
    defs = {
        ('r2', 'r-squared', 'r²'): "R² (R-squared) measures how much variance in the target the model explains. 1.0 = perfect, 0.0 = no better than predicting the mean. Above 0.7 is generally strong.",
        ('rmse',): "RMSE is the square root of the average squared error. It is in the same units as the target and penalises large mistakes more than MAE. Lower = better.",
        ('mae',): "MAE (Mean Absolute Error) is the average of absolute differences between actual and predicted values — easy to interpret as the average mistake the model makes.",
        ('f1',): "F1 score is the harmonic mean of precision and recall (0–1). It is ideal when classes are imbalanced, balancing false positives and false negatives.",
        ('precision',): "Precision = of all samples predicted as positive, what fraction were truly positive. High precision = few false alarms.",
        ('recall',): "Recall = of all actual positives, what fraction did the model correctly identify. High recall = few missed detections.",
        ('overfitting',): "Overfitting happens when a model memorises training data but performs poorly on new data. Signs: very high training accuracy but low test accuracy. Use regularisation or cross-validation.",
        ('underfitting',): "Underfitting means the model is too simple to capture patterns — both training and test accuracy are low. Try more complex models or add informative features.",
    }
    for keys, ans in defs.items():
        if any(k in q for k in keys):
            return ans

    if any(x in q for x in ['row', 'column', 'shape', 'size', 'how many', 'dataset']):
        return (f"The dataset has {intake['rows']:,} rows and {intake['cols']} columns. "
                f"Numeric ({len(intake['numeric'])}): {', '.join(intake['numeric'][:8]) or 'none'}. "
                f"Categorical ({len(intake['categorical'])}): {', '.join(intake['categorical'][:5]) or 'none'}. "
                f"Completeness: {intake['completeness']}%.")

    if any(x in q for x in ['missing', 'null', 'nan', 'completeness']):
        if intake['missing'] == 0:
            return f"No missing values — all {intake['rows']*intake['cols']:,} cells are filled."
        return (f"{intake['missing']} missing values detected ({100-intake['completeness']:.1f}% of cells). "
                "The preprocessing pipeline automatically imputes numeric columns with the median "
                "and categorical columns with the mode.")

    if any(x in q for x in ['improve', 'better', 'boost', 'enhance', 'tip', 'recommend']):
        tips = []
        if ml:
            score = ml['accuracy'] if task == 'classification' else ml['r2']
            if score < 0.7:
                tips.append("try Gradient Boosting or XGBoost for stronger performance")
                tips.append("consider feature engineering — polynomial features, interaction terms")
            else:
                tips.append("tune hyperparameters with GridSearchCV")
        if intake['missing'] > 0:
            tips.append(f"investigate the {intake['missing']} missing values and their cause")
        tips.append("use k-fold cross-validation (k=5 or 10) for more reliable metrics")
        return "Improvement suggestions: " + "; ".join(tips) + "."

    # Fallback
    if ml:
        if task == 'classification':
            return (f"Dataset: {intake['rows']:,} rows, {intake['cols']} cols, {intake['completeness']}% complete. "
                    f"Best classifier: {ml['best']} — Accuracy={ml['accuracy']:.4f}, F1={ml['f1']:.4f}, "
                    f"target='{ml['target']}' ({ml['n_classes']} classes). "
                    "Ask about accuracy, F1, class balance, feature importance, correlations, or patterns.")
        return (f"Dataset: {intake['rows']:,} rows, {intake['cols']} cols, {intake['completeness']}% complete. "
                f"Best model: {ml['best']} — R²={ml['r2']:.4f}, RMSE={ml['rmse']:.4f}, "
                f"target='{ml['target']}'. "
                "Ask about R², RMSE, feature importance, patterns, or correlations.")
    return (f"Dataset: {intake['rows']:,} rows, {intake['cols']} cols, {intake['completeness']}% complete. "
            "No models trained yet. Ask about dataset size, columns, missing values, or correlations.")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ◆ DataAnalytica")
    st.markdown(
        "<p style='color:#78716c;font-size:12px;margin-top:-8px;margin-bottom:16px'>"
        "Multi-Agent Intelligent Data Analytics System</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Upload
    st.markdown("<p style='color:#57534e;font-size:13px;font-weight:600;letter-spacing:.5px;"
                "text-transform:uppercase;margin-bottom:6px'>Upload Dataset</p>",
                unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    st.markdown("<p style='color:#57534e;font-size:13px;margin-top:-6px'>Any comma-separated dataset</p>",
                unsafe_allow_html=True)

    if st.session_state.df is not None:
        intake_s = st.session_state.analysis['intake']
        all_cols  = st.session_state.df.columns.tolist()
        st.markdown("---")

        # Target column
        st.markdown("<p style='color:#57534e;font-size:13px;font-weight:600;letter-spacing:.5px;"
                    "text-transform:uppercase;margin-bottom:6px'>Target Variable</p>",
                    unsafe_allow_html=True)
        default_t = st.session_state.selected_target or infer_target(st.session_state.df)
        t_idx = all_cols.index(default_t) if default_t in all_cols else 0
        sel_target = st.selectbox("Column to predict", options=all_cols if all_cols else ['—'],
                                  index=t_idx, key="target_sel")

        # Task detection
        if sel_target and sel_target in st.session_state.df.columns:
            detected = detect_task(st.session_state.df, sel_target)
            det_color = '#16a34a' if detected == 'classification' else '#2563eb'
            st.markdown(f"<p style='color:{det_color};font-size:13px;margin-top:-4px'>"
                        f"→ Auto-detected: <strong>{detected}</strong></p>",
                        unsafe_allow_html=True)
            cur_task = detected
        else:
            cur_task = 'regression'

        # Task override
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='color:#57534e;font-size:13px;font-weight:600;letter-spacing:.5px;"
                    "text-transform:uppercase;margin-bottom:6px'>Task Override</p>",
                    unsafe_allow_html=True)
        task_override = st.selectbox("Select task", ["Auto Detect", "Classification", "Regression"],
                                     key="task_override_sel")
        if task_override == "Classification":
            cur_task = 'classification'
        elif task_override == "Regression":
            cur_task = 'regression'

        # Models
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='color:#57534e;font-size:13px;font-weight:600;letter-spacing:.5px;"
                    "text-transform:uppercase;margin-bottom:6px'>Models</p>",
                    unsafe_allow_html=True)
        if cur_task == 'classification':
            model_pool = CLF_MODELS
            default_on = {"Logistic Regression", "Random Forest", "Gradient Boosting", "Naive Bayes"}
        else:
            model_pool = REG_MODELS
            default_on = {"Linear Regression", "Random Forest", "Gradient Boosting"}
        model_choices = {}
        for name in model_pool:
            model_choices[name] = st.checkbox(name, value=(name in default_on), key=f"chk_{name}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶  Run Models", use_container_width=True):
            chosen = [n for n, v in model_choices.items() if v]
            if not chosen:
                st.warning("Select at least one model.")
            elif not sel_target or sel_target not in st.session_state.df.columns:
                st.warning("Please select a valid target column.")
            else:
                st.session_state.selected_target  = sel_target
                st.session_state.selected_models  = chosen
                st.session_state.models_trained   = False
                st.session_state.task_type        = cur_task
                st.rerun()

        # Agent status
        st.markdown("---")
        st.markdown("<p style='color:#57534e;font-size:13px;font-weight:600;letter-spacing:.5px;"
                    "text-transform:uppercase;margin-bottom:4px'>Agent Status</p>",
                    unsafe_allow_html=True)
        icons  = {'done': '✓', 'running': '◌', 'idle': '○'}
        colors = {'done': '#16a34a', 'running': '#ea580c', 'idle': '#57534e'}
        agent_labels = [
            ('intake',     'Intake Agent'),
            ('preprocess', 'Preprocessing Agent'),
            ('eda',        'EDA Agent'),
            ('viz',        'Visualization Agent'),
            ('ml',         'ML Agent'),
            ('diagnostic', 'Diagnostic Agent'),
            ('insight',    'Insight Agent'),
        ]
        for agent, label in agent_labels:
            s = st.session_state.agent_status.get(agent, 'idle')
            st.markdown(
                f"<p style='color:{colors[s]};font-size:13px;margin:3px 0'>"
                f"{icons[s]}  {label}</p>",
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING TRIGGER
# ══════════════════════════════════════════════════════════════════════════════
if uploaded is not None:
    file_id = uploaded.name + str(uploaded.size)
    if st.session_state.last_file != file_id:
        df_new = load_data(uploaded)
        st.session_state.df             = df_new
        st.session_state.last_file      = file_id
        st.session_state.chat_history   = []
        st.session_state.ml_result      = None
        st.session_state.models_trained = False
        st.session_state.agent_log      = []
        tgt = infer_target(df_new)
        st.session_state.selected_target = tgt
        st.session_state.task_type       = detect_task(df_new, tgt or '')
        st.session_state.agent_status    = {
            a: 'idle' for a in ['intake', 'preprocess', 'eda', 'viz', 'ml', 'diagnostic', 'insight']
        }
        with st.spinner("Loading dataset…"):
            st.session_state.analysis = run_initial_pipeline(df_new)
        st.rerun()

if (st.session_state.df is not None and
        st.session_state.selected_models and
        not st.session_state.models_trained):
    with st.spinner("Training models…"):
        st.session_state.agent_status['ml'] = 'running'
        ml_res = train_models(
            st.session_state.df,
            st.session_state.selected_target,
            st.session_state.selected_models
        )
        st.session_state.ml_result      = ml_res
        st.session_state.models_trained = True
        st.session_state.agent_status['ml'] = 'done'

        st.session_state.agent_status['diagnostic'] = 'running'
        diags = generate_diagnostics(st.session_state.df, ml_res)
        st.session_state.analysis['diagnostics'] = diags
        st.session_state.agent_status['diagnostic'] = 'done'

        st.session_state.agent_status['insight'] = 'running'
        st.session_state.analysis['insights'] = get_insights(st.session_state.df, ml_res)
        st.session_state.analysis['ml']       = ml_res
        st.session_state.agent_status['insight'] = 'done'
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.df is None:
    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px'>
            <div style='font-size:40px;margin-bottom:16px;color:#1c1917'>◆</div>
            <h1 style='font-size:28px;margin-bottom:12px'>DataAnalytica</h1>
            <p style='color:#78716c;font-size:15px;font-style:italic;margin-bottom:4px'>
                Multi-Agent Intelligent Data Analytics System
            </p>
            <p style='color:#78716c;font-size:16px;line-height:1.8;max-width:440px;margin:16px auto 32px'>
                Upload a CSV from the sidebar. Seven specialised agents will analyse your data —
                from schema detection through automatic preprocessing, machine learning,
                diagnostics, and plain-language insights.
            </p>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:440px;margin:0 auto;text-align:left'>
                <div style='background:#fff;border:1px solid #e7e5e4;border-radius:9px;padding:15px 18px'>
                    <div style='font-size:14px;font-weight:600;color:#1c1917;margin-bottom:4px'>Intake Agent</div>
                    <div style='font-size:13px;color:#78716c'>Schema, types, completeness</div>
                </div>
                <div style='background:#fff;border:1px solid #e7e5e4;border-radius:9px;padding:15px 18px'>
                    <div style='font-size:14px;font-weight:600;color:#1c1917;margin-bottom:4px'>Preprocessing Agent</div>
                    <div style='font-size:13px;color:#78716c'>Auto impute, encode, scale</div>
                </div>
                <div style='background:#fff;border:1px solid #e7e5e4;border-radius:9px;padding:15px 18px'>
                    <div style='font-size:14px;font-weight:600;color:#1c1917;margin-bottom:4px'>EDA Agent</div>
                    <div style='font-size:13px;color:#78716c'>Statistics and correlations</div>
                </div>
                <div style='background:#fff;border:1px solid #e7e5e4;border-radius:9px;padding:15px 18px'>
                    <div style='font-size:14px;font-weight:600;color:#1c1917;margin-bottom:4px'>Visualization Agent</div>
                    <div style='font-size:13px;color:#78716c'>Interactive Plotly charts</div>
                </div>
                <div style='background:#fff;border:1px solid #e7e5e4;border-radius:9px;padding:15px 18px'>
                    <div style='font-size:14px;font-weight:600;color:#1c1917;margin-bottom:4px'>ML Agent</div>
                    <div style='font-size:13px;color:#78716c'>Auto regression or classification</div>
                </div>
                <div style='background:#fff;border:1px solid #e7e5e4;border-radius:9px;padding:15px 18px'>
                    <div style='font-size:14px;font-weight:600;color:#1c1917;margin-bottom:4px'>Diagnostic Agent</div>
                    <div style='font-size:13px;color:#78716c'>Why accuracy is low + fixes</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PULL STATE
# ══════════════════════════════════════════════════════════════════════════════
df       = st.session_state.df
analysis = st.session_state.analysis
intake   = analysis['intake']
eda      = analysis['eda']
ml       = analysis.get('ml') or st.session_state.ml_result
insights = analysis.get('insights') or []
diags    = analysis.get('diagnostics') or []
fname    = uploaded.name if uploaded else "dataset"
task     = (ml.get('task') if ml else None) or st.session_state.task_type
target   = st.session_state.selected_target or infer_target(df)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<h1 style='margin-bottom:2px'>◆ DataAnalytica</h1>"
    f"<p style='color:#78716c;font-size:13px;font-style:italic;margin-top:0;margin-bottom:4px'>"
    f"Multi-Agent Intelligent Data Analytics System</p>"
    f"<p style='color:#78716c;font-size:15px;margin-top:0;margin-bottom:20px'>"
    f"{fname} &nbsp;·&nbsp; {intake['rows']:,} rows &nbsp;·&nbsp; "
    f"{intake['cols']} columns &nbsp;·&nbsp; {intake['completeness']}% complete</p>",
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows",         f"{intake['rows']:,}")
c2.metric("Columns",      str(intake['cols']), f"{len(intake['numeric'])} numeric")
c3.metric("Completeness", f"{intake['completeness']}%",
          "no missing" if intake['missing'] == 0 else f"{intake['missing']} missing")
if ml:
    if task == 'classification':
        c4.metric("Best Accuracy", f"{ml['accuracy']:.4f}", ml['best'])
        c5.metric("Best F1",       f"{ml['f1']:.4f}", f"target: {ml['target']}")
    else:
        c4.metric("Best R²",   f"{ml['r2']:.4f}",  ml['best'])
        c5.metric("Best RMSE", f"{ml['rmse']:.4f}", f"target: {ml['target']}")
else:
    c4.metric("ML Status", "Pending", "choose models in sidebar")
    c5.metric("Target",    target or "—")

st.markdown("<br>", unsafe_allow_html=True)

# 6 tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dataset Overview",
    "📊 Visualizations",
    "🤖 ML Models",
    "🔬 Model Diagnostics",
    "💡 AI Insights",
    "💬 Ask Questions"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([1.3, 1])

    with col_l:
        st.markdown("### Agent Execution Log")
        if ml:
            if task == 'classification':
                ml_log = (f"Trained {len(ml['models'])} classifier(s) on '{ml['target']}'. "
                          f"Best: {ml['best']} — Accuracy={ml['accuracy']:.3f}, F1={ml['f1']:.3f}.")
            else:
                ml_log = (f"Trained {len(ml['models'])} model(s) on '{ml['target']}'. "
                          f"Best: {ml['best']} — R²={ml['r2']:.3f}, RMSE={ml['rmse']:.3f}.")
        else:
            ml_log = "Waiting — select models in the sidebar and click Run Models."

        n_nc = len(eda['ncols'])
        logs = [
            ("Intake Agent",
             f"Loaded {intake['rows']:,} rows and {intake['cols']} columns. "
             f"Found {len(intake['numeric'])} numeric and {len(intake['categorical'])} categorical features. "
             f"Completeness: {intake['completeness']}% "
             f"({'no missing values' if intake['missing'] == 0 else str(intake['missing']) + ' missing values'}).",
             "step 1", C['teal']),
            ("Preprocessing Agent",
             "Automatic pipeline built: numeric → mean/median imputation + StandardScaler; "
             "categorical → mode imputation + OneHotEncoder. Handles messy, mixed-type data.",
             "step 2", C['pink']),
            ("EDA Agent",
             f"Computed descriptive statistics for {len(intake['numeric'])} numeric columns. "
             f"Built a {n_nc}×{n_nc} correlation matrix.",
             "step 3", C['blue']),
            ("Visualization Agent",
             "Prepared correlation heatmap, missing-value map, distributions, pair plots, scatter, and boxplots.",
             "step 4", C['orange']),
            ("ML Agent", ml_log, "on demand", C['purple']),
            ("Diagnostic Agent",
             f"Generated {len(diags)} diagnostic findings." if diags else "Will run after ML models are trained.",
             "after ML", C['red']),
            ("Insight Agent",
             f"Generated {len(insights)} findings based on data statistics and ML results."
             if insights else "Will run after ML models are trained.",
             "after ML", C['green']),
        ]
        log_html = "<div style='background:#fff;border:1px solid #e7e5e4;border-radius:10px;overflow:hidden'>"
        for i, (agent, msg, t, clr) in enumerate(logs):
            sep = "border-bottom:1px solid #f5f5f4;" if i < len(logs) - 1 else ""
            log_html += f"""<div style='display:flex;gap:14px;padding:15px 20px;{sep}'>
                <div style='width:3px;border-radius:2px;background:{clr};flex-shrink:0;margin:2px 0'></div>
                <div style='flex:1'>
                    <div style='font-size:14px;font-weight:600;color:#1c1917;margin-bottom:4px'>{agent}</div>
                    <div style='font-size:13px;color:#78716c;line-height:1.65'>{msg}</div>
                </div>
                <div style='font-size:12px;color:#a8a29e;font-family:JetBrains Mono,monospace;
                            padding-top:2px;white-space:nowrap'>{t}</div>
            </div>"""
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)

        # Live agent log
        if st.session_state.agent_log:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Live Agent Log")
            log_lines = "\n".join(st.session_state.agent_log[-30:])
            st.code(log_lines, language=None)

    with col_r:
        st.markdown("### Column Profile")
        rows_html = ""
        for col in df.columns[:18]:
            is_num = col in intake['numeric']
            bg_col = "#eff6ff" if is_num else "#fff7ed"
            tx_col = "#1d4ed8" if is_num else "#c2410c"
            uniq   = df[col].nunique()
            nulls  = int(df[col].isnull().sum())
            null_s = f" · {nulls} null" if nulls > 0 else ""
            rows_html += f"""<div style='display:flex;align-items:center;justify-content:space-between;
                        padding:9px 0;border-bottom:1px solid #f5f5f4'>
                <span style='color:#1c1917;font-family:JetBrains Mono,monospace;font-size:13px;
                             max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{col}</span>
                <div style='display:flex;align-items:center;gap:8px;flex-shrink:0'>
                    <span style='color:#a8a29e;font-size:12px'>{uniq} uniq{null_s}</span>
                    <span style='background:{bg_col};color:{tx_col};padding:3px 10px;
                                 border-radius:20px;font-size:12px;font-weight:500'>
                        {'numeric' if is_num else 'text'}</span>
                </div>
            </div>"""
        if len(df.columns) > 18:
            rows_html += f"<p style='color:#a8a29e;font-size:13px;padding:6px 0 0'>+{len(df.columns)-18} more columns</p>"
        st.markdown(f"<div style='background:#fff;border:1px solid #e7e5e4;border-radius:10px;"
                    f"padding:16px 20px'>{rows_html}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Data Preview")
    st.dataframe(df.head(8), use_container_width=True)
    if eda['stats'] is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Descriptive Statistics")
        st.dataframe(eda['stats'].round(4), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    generate_visualizations(df, eda, target)

    # ── Model Insights section (only when ML done) ────────────────────────────
    if ml:
        st.markdown("---")
        st.markdown("## 🧠 Model Insights")
        c_mi1, c_mi2 = st.columns(2)
        best_res = ml['models'][ml['best']]

        with c_mi1:
            if best_res.get('importance'):
                st.markdown(f"### Feature Importance — {ml['best']}")
                si = sorted(best_res['importance'].items(), key=lambda x: x[1], reverse=True)[:12]
                fig_ic = go.Figure(go.Bar(
                    x=[v for _, v in si], y=[n[:24] for n, _ in si], orientation='h',
                    marker_color=C['purple'], opacity=0.85,
                    text=[f"{v:.4f}" for _, v in si], textposition='outside',
                    textfont=dict(size=12, color='#44403c')
                ))
                _pl_ic = {k: v for k, v in PL.items() if k != 'yaxis'}
                fig_ic.update_layout(
                    **_pl_ic, height=max(280, len(si) * 32),
                    yaxis=dict(**PL['yaxis'], autorange='reversed'),
                    showlegend=False, xaxis_title="Importance score"
                )
                st.plotly_chart(fig_ic, use_container_width=True)

        with c_mi2:
            if task == 'classification':
                st.markdown("### Classifier Accuracy Comparison")
                mn = list(ml['models'].keys())
                accs = [ml['models'][k]['accuracy'] for k in mn]
                max_a = max(accs) if accs else 0
                fig_cm = go.Figure(go.Bar(
                    x=mn, y=accs,
                    marker_color=[C['green'] if a == max_a else C['blue'] for a in accs],
                    text=[f"{a:.4f}" for a in accs], textposition='outside',
                    textfont=dict(color='#44403c', size=12)
                ))
                _pl2 = {k: v for k, v in PL.items() if k != 'xaxis'}
                fig_cm.update_layout(
                    **_pl2, yaxis_range=[0, min(1.0, max_a * 1.2 + 0.05)],
                    showlegend=False, height=320,
                    xaxis=dict(**PL['xaxis'], tickangle=-25)
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.markdown("### Regressor R² Comparison")
                mnr = list(ml['models'].keys())
                r2s = [ml['models'][k]['r2'] for k in mnr]
                max_r = max(r2s) if r2s else 0
                fig_rm = go.Figure(go.Bar(
                    x=mnr, y=r2s,
                    marker_color=[C['green'] if r == max_r else C['primary'] for r in r2s],
                    text=[f"{r:.4f}" for r in r2s], textposition='outside',
                    textfont=dict(color='#44403c', size=12)
                ))
                _pl3 = {k: v for k, v in PL.items() if k != 'xaxis'}
                fig_rm.update_layout(
                    **_pl3, yaxis_range=[0, max(max_r * 1.3 + 0.05, 0.2)],
                    showlegend=False, height=320,
                    xaxis=dict(**PL['xaxis'], tickangle=-25)
                )
                st.plotly_chart(fig_rm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not ml:
        st.markdown(
            "<div style='background:#fff;border:1px solid #e7e5e4;border-radius:10px;"
            "padding:32px;text-align:center;margin-top:20px'>"
            "<p style='font-size:16px;font-weight:600;color:#1c1917;margin-bottom:8px'>"
            "No models trained yet</p>"
            "<p style='font-size:15px;color:#78716c;max-width:440px;margin:0 auto'>"
            "Select a target variable and models in the sidebar, then click "
            "<strong>▶ Run Models</strong>. Task (regression or classification) is auto-detected.</p>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        if task == 'classification':
            # ── CLASSIFICATION ────────────────────────────────────────────────
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Target",     ml['target'])
            m2.metric("Task",       "Classification")
            m3.metric("Best Model", ml['best'])
            m4.metric("Accuracy",   f"{ml['accuracy']:.4f}", f"{ml['accuracy']*100:.1f}%")
            m5.metric("F1 Score",   f"{ml['f1']:.4f}")
            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("About the selected models"):
                desc_html = "<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:4px 0'>"
                for name in ml['models']:
                    desc_html += (f"<div style='background:#faf9f7;border:1px solid #e7e5e4;"
                                  f"border-radius:8px;padding:14px 16px'>"
                                  f"<div style='font-size:14px;font-weight:600;color:#1c1917;"
                                  f"margin-bottom:5px'>{name}</div>"
                                  f"<div style='font-size:13px;color:#78716c;line-height:1.6'>"
                                  f"{MODEL_DESC.get(name, '')}</div></div>")
                desc_html += "</div>"
                st.markdown(desc_html, unsafe_allow_html=True)

            col_l3, col_r3 = st.columns(2)
            with col_l3:
                st.markdown("### Accuracy Comparison")
                mn  = list(ml['models'].keys())
                accs = [ml['models'][k]['accuracy'] for k in mn]
                max_acc = max(accs) if accs else 0.0
                fig_bc = go.Figure(go.Bar(
                    x=mn, y=accs,
                    marker_color=[C['green'] if a == max_acc else C['blue'] for a in accs],
                    text=[f"{a:.4f}" for a in accs], textposition='outside',
                    textfont=dict(color='#44403c', size=13)
                ))
                _pl_bc = {k: v for k, v in PL.items() if k != 'xaxis'}
                fig_bc.update_layout(**_pl_bc, yaxis_range=[0, min(1.0, max_acc * 1.2 + 0.05)],
                                     showlegend=False, height=320,
                                     xaxis=dict(**PL['xaxis'], tickangle=-25))
                st.plotly_chart(fig_bc, use_container_width=True)

            with col_r3:
                st.markdown("### Performance Report")
                rr = [{'Model': n + (" ★" if n == ml['best'] else ""),
                       'Accuracy': f"{r['accuracy']:.4f}", 'Precision': f"{r['precision']:.4f}",
                       'Recall': f"{r['recall']:.4f}", 'F1': f"{r['f1']:.4f}"}
                      for n, r in ml['models'].items()]
                st.dataframe(pd.DataFrame(rr).set_index('Model'), use_container_width=True)
                acc_label = ("strong — highly reliable" if ml['accuracy'] > 0.85
                             else "moderate — some classes may be confused" if ml['accuracy'] > 0.65
                             else "needs improvement — see Diagnostics tab")
                st.markdown(f"""<div style='background:#f0fdf4;border:1px solid #e7e5e4;
                    border-left:3px solid {C['green']};border-radius:9px;padding:16px 18px;margin-top:14px'>
                    <div style='font-size:12px;font-weight:600;color:{C['green']};letter-spacing:.8px;
                                text-transform:uppercase;margin-bottom:8px'>Best — {ml['best']}</div>
                    <p style='font-size:14px;color:#44403c;line-height:1.75;margin:0'>
                        Accuracy <strong>{ml['accuracy']:.4f}</strong> is {acc_label}.
                        F1=<strong>{ml['f1']:.4f}</strong>, Precision=<strong>{ml['precision']:.4f}</strong>,
                        Recall=<strong>{ml['recall']:.4f}</strong>.
                        Classes: <em>{', '.join(str(c) for c in ml.get('classes', [])[:8])}</em>.
                        Trained on {ml['n_train']} samples, tested on {ml['n_test']}.
                    </p></div>""", unsafe_allow_html=True)

            if ml.get('report'):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Full Classification Report")
                st.code(ml['report'], language=None)

            best_res_c = ml['models'][ml['best']]
            if best_res_c.get('importance'):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"### Feature Importance — {ml['best']}")
                si = sorted(best_res_c['importance'].items(), key=lambda x: x[1], reverse=True)[:12]
                fig_ic = go.Figure(go.Bar(
                    x=[v for _, v in si], y=[n[:24] for n, _ in si], orientation='h',
                    marker_color=C['purple'], opacity=0.85,
                    text=[f"{v:.4f}" for _, v in si], textposition='outside',
                    textfont=dict(size=12, color='#44403c')
                ))
                _pl_ic = {k: v for k, v in PL.items() if k != 'yaxis'}
                fig_ic.update_layout(**_pl_ic, height=max(280, len(si) * 32),
                                     yaxis=dict(**PL['yaxis'], autorange='reversed'),
                                     showlegend=False, xaxis_title="Importance score")
                st.plotly_chart(fig_ic, use_container_width=True)

            y_tc = best_res_c.get('y_test', [])
            p_tc = best_res_c.get('preds', [])
            n_c  = min(len(y_tc), len(p_tc))
            if n_c > 0:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Actual vs Predicted Labels")
                fig_pc = go.Figure()
                fig_pc.add_trace(go.Scatter(x=list(range(n_c)), y=y_tc[:n_c], mode='markers',
                                            name='Actual', marker=dict(color=C['orange'], size=7, opacity=0.8)))
                fig_pc.add_trace(go.Scatter(x=list(range(n_c)), y=p_tc[:n_c], mode='lines',
                                            name='Predicted', line=dict(color=C['blue'], width=2, dash='dot')))
                fig_pc.update_layout(**PL, height=300,
                                     legend=dict(font=dict(color='#44403c', size=13), bgcolor='rgba(0,0,0,0)'),
                                     xaxis_title="Sample index", yaxis_title=ml['target'])
                st.plotly_chart(fig_pc, use_container_width=True)

        else:
            # ── REGRESSION ────────────────────────────────────────────────────
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Target",     ml['target'])
            m2.metric("Models run", str(len(ml['models'])))
            m3.metric("Best model", ml['best'])
            m4.metric("Best R²",    f"{ml['r2']:.4f}", f"{ml['r2']*100:.1f}% explained")
            m5.metric("Best RMSE",  f"{ml['rmse']:.4f}")
            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("About the selected models"):
                desc_html = "<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:4px 0'>"
                for name in ml['models']:
                    desc_html += (f"<div style='background:#faf9f7;border:1px solid #e7e5e4;"
                                  f"border-radius:8px;padding:14px 16px'>"
                                  f"<div style='font-size:14px;font-weight:600;color:#1c1917;"
                                  f"margin-bottom:5px'>{name}</div>"
                                  f"<div style='font-size:13px;color:#78716c;line-height:1.6'>"
                                  f"{MODEL_DESC.get(name, '')}</div></div>")
                desc_html += "</div>"
                st.markdown(desc_html, unsafe_allow_html=True)

            col_l3, col_r3 = st.columns(2)
            with col_l3:
                st.markdown("### R² Comparison")
                mnr = list(ml['models'].keys())
                r2s = [ml['models'][k]['r2'] for k in mnr]
                max_r2 = max(r2s) if r2s else 0.0
                fig_bar = go.Figure(go.Bar(
                    x=mnr, y=r2s,
                    marker_color=[C['green'] if r == max_r2 else C['primary'] for r in r2s],
                    text=[f"{r:.4f}" for r in r2s], textposition='outside',
                    textfont=dict(color='#44403c', size=13)
                ))
                _pl_bar = {k: v for k, v in PL.items() if k != 'xaxis'}
                fig_bar.update_layout(**_pl_bar, yaxis_range=[0, max(max_r2 * 1.3 + 0.05, 0.2)],
                                      showlegend=False, height=320,
                                      xaxis=dict(**PL['xaxis'], tickangle=-25))
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_r3:
                st.markdown("### Performance Report")
                rr = [{'Model': n + (" ★" if n == ml['best'] else ""),
                       'R²': f"{r['r2']:.4f}", 'MAE': f"{r['mae']:.4f}", 'RMSE': f"{r['rmse']:.4f}"}
                      for n, r in ml['models'].items()]
                st.dataframe(pd.DataFrame(rr).set_index('Model'), use_container_width=True)
                r2_label = ("strong — captures most variance" if ml['r2'] > 0.7
                            else "moderate — real signal, some variance unexplained" if ml['r2'] > 0.4
                            else "weak — see Diagnostics tab for improvement tips")
                st.markdown(f"""<div style='background:#f0fdf4;border:1px solid #e7e5e4;
                    border-left:3px solid {C['green']};border-radius:9px;padding:16px 18px;margin-top:14px'>
                    <div style='font-size:12px;font-weight:600;color:{C['green']};letter-spacing:.8px;
                                text-transform:uppercase;margin-bottom:8px'>Best — {ml['best']}</div>
                    <p style='font-size:14px;color:#44403c;line-height:1.75;margin:0'>
                        R² of <strong>{ml['r2']:.4f}</strong> is {r2_label}.
                        MAE=<strong>{ml['mae']:.4f}</strong>, RMSE=<strong>{ml['rmse']:.4f}</strong>
                        (units of <em>{ml['target']}</em>).
                        Trained on {ml['n_train']} samples, tested on {ml['n_test']}.
                    </p></div>""", unsafe_allow_html=True)

            best_res = ml['models'][ml['best']]
            if best_res.get('importance'):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"### Feature Importance — {ml['best']}")
                si = sorted(best_res['importance'].items(), key=lambda x: x[1], reverse=True)[:12]
                fig_imp = go.Figure(go.Bar(
                    x=[v for _, v in si], y=[n[:24] for n, _ in si], orientation='h',
                    marker_color=C['blue'], opacity=0.85,
                    text=[f"{v:.4f}" for _, v in si], textposition='outside',
                    textfont=dict(size=12, color='#44403c')
                ))
                _pl_imp = {k: v for k, v in PL.items() if k != 'yaxis'}
                fig_imp.update_layout(**_pl_imp, height=max(260, len(si) * 32),
                                      yaxis=dict(**PL['yaxis'], autorange='reversed'),
                                      showlegend=False, xaxis_title="Importance score")
                st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Actual vs Predicted Values")
            y_tb = best_res['y_test']
            p_b  = best_res['preds']
            n_pts = min(len(y_tb), len(p_b))
            if n_pts > 0:
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=list(range(n_pts)), y=y_tb[:n_pts],
                                              mode='markers', name='Actual',
                                              marker=dict(color=C['orange'], size=6, opacity=0.8)))
                fig_pred.add_trace(go.Scatter(x=list(range(n_pts)), y=p_b[:n_pts],
                                              mode='lines', name='Predicted',
                                              line=dict(color=C['primary'], width=2, dash='dot')))
                fig_pred.update_layout(**PL, height=300,
                                       legend=dict(font=dict(color='#44403c', size=13),
                                                   bgcolor='rgba(0,0,0,0)'),
                                       xaxis_title="Sample index", yaxis_title=ml['target'])
                st.plotly_chart(fig_pred, use_container_width=True)

            st.markdown("### Residual Analysis")
            if n_pts > 0:
                residuals = [a - p for a, p in zip(y_tb[:n_pts], p_b[:n_pts])]
                fig_res = go.Figure(go.Scatter(
                    x=p_b[:n_pts], y=residuals, mode='markers',
                    marker=dict(color=C['teal'], size=5, opacity=0.65)
                ))
                fig_res.add_hline(y=0, line_color=C['red'], line_dash='dash', line_width=1.5)
                fig_res.update_layout(**PL, height=280,
                                      xaxis_title="Predicted values",
                                      yaxis_title="Residual (actual − predicted)",
                                      showlegend=False)
                st.plotly_chart(fig_res, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🔬 Model Diagnostic Report")
    st.markdown(
        "<p style='color:#78716c;font-size:15px;margin-top:-6px;margin-bottom:20px'>"
        "Automated analysis of why performance may be low, with specific improvement suggestions.</p>",
        unsafe_allow_html=True
    )

    if not diags:
        st.markdown(
            "<div style='background:#fff;border:1px solid #e7e5e4;border-radius:10px;"
            "padding:28px;text-align:center'>"
            "<p style='color:#78716c;font-size:15px;margin:0'>"
            "Diagnostics appear here after training models from the sidebar.</p></div>",
            unsafe_allow_html=True
        )
    else:
        sev_cfg = {
            'success': {'color': C['green'],  'bg': '#f0fdf4', 'label': 'OK'},
            'warning': {'color': C['amber'],  'bg': '#fffbeb', 'label': 'Warning'},
            'info':    {'color': C['blue'],   'bg': '#eff6ff', 'label': 'Info'},
        }
        for d in diags:
            cfg = sev_cfg.get(d.get('severity', 'info'), sev_cfg['info'])
            sugg_html = ""
            if d.get('suggestion'):
                sugg_html = (f"<div style='margin-top:10px;padding:10px 14px;background:rgba(0,0,0,0.03);"
                             f"border-radius:6px;font-size:13px;color:#44403c'>"
                             f"<strong>💡 Suggestion:</strong> {d['suggestion']}</div>")
            st.markdown(f"""<div style='background:{cfg["bg"]};border:1px solid #e7e5e4;
                        border-left:3px solid {cfg["color"]};border-radius:9px;
                        padding:18px 20px;margin-bottom:13px'>
                <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:8px'>
                    <span style='font-size:15px;font-weight:600;color:#1c1917'>
                        {d.get("icon","")} {d.get("title","")}</span>
                    <span style='font-size:11px;font-weight:600;color:{cfg["color"]};
                                 background:rgba(0,0,0,0.04);padding:3px 10px;border-radius:20px;
                                 letter-spacing:.5px'>{cfg["label"].upper()}</span>
                </div>
                <p style='font-size:14px;color:#44403c;line-height:1.8;margin:0'>{d.get("body","")}</p>
                {sugg_html}
            </div>""", unsafe_allow_html=True)

    # Preprocessing summary
    st.markdown("---")
    st.markdown("### Preprocessing Summary")
    num_cols_list = df.select_dtypes(include='number').columns.tolist()
    cat_cols_list = df.select_dtypes(exclude='number').columns.tolist()
    miss_cols = {c: int(df[c].isnull().sum()) for c in df.columns if df[c].isnull().sum() > 0}
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown(f"""<div style='background:#fff;border:1px solid #e7e5e4;border-radius:9px;padding:16px 18px'>
            <div style='font-size:12px;font-weight:600;color:#78716c;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px'>Numeric Features</div>
            <div style='font-size:22px;font-weight:600;color:#1c1917;margin-bottom:6px'>{len(num_cols_list)}</div>
            <div style='font-size:13px;color:#78716c'>Imputed with median → StandardScaler applied</div>
        </div>""", unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"""<div style='background:#fff;border:1px solid #e7e5e4;border-radius:9px;padding:16px 18px'>
            <div style='font-size:12px;font-weight:600;color:#78716c;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px'>Categorical Features</div>
            <div style='font-size:22px;font-weight:600;color:#1c1917;margin-bottom:6px'>{len(cat_cols_list)}</div>
            <div style='font-size:13px;color:#78716c'>Imputed with mode → OneHotEncoded</div>
        </div>""", unsafe_allow_html=True)
    with col_p3:
        st.markdown(f"""<div style='background:#fff;border:1px solid #e7e5e4;border-radius:9px;padding:16px 18px'>
            <div style='font-size:12px;font-weight:600;color:#78716c;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px'>Columns with Missing</div>
            <div style='font-size:22px;font-weight:600;color:#1c1917;margin-bottom:6px'>{len(miss_cols)}</div>
            <div style='font-size:13px;color:#78716c'>{"None — dataset is complete" if not miss_cols else ", ".join(f"{k}:{v}" for k, v in list(miss_cols.items())[:4])}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Key Findings")
    if not insights:
        st.markdown(
            "<div style='background:#fff;border:1px solid #e7e5e4;border-radius:10px;"
            "padding:28px;text-align:center'>"
            "<p style='color:#78716c;font-size:15px;margin:0'>"
            "Insights appear here after running models from the sidebar.</p></div>",
            unsafe_allow_html=True
        )
    else:
        type_cfg = {
            'info':    {'color': C['blue'],   'bg': '#eff6ff', 'label': 'Note'},
            'success': {'color': C['green'],  'bg': '#f0fdf4', 'label': 'Good'},
            'warning': {'color': C['amber'],  'bg': '#fffbeb', 'label': 'Attention'},
            'trend':   {'color': C['teal'],   'bg': '#f0fdfa', 'label': 'Trend'},
        }
        for ins in insights:
            cfg = type_cfg.get(ins.get('type', 'info'), type_cfg['info'])
            st.markdown(f"""<div style='background:{cfg["bg"]};border:1px solid #e7e5e4;
                        border-left:3px solid {cfg["color"]};border-radius:9px;
                        padding:18px 20px;margin-bottom:13px'>
                <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:8px'>
                    <span style='font-size:15px;font-weight:600;color:#1c1917'>{ins.get("title","")}</span>
                    <span style='font-size:11px;font-weight:600;color:{cfg["color"]};background:rgba(0,0,0,0.04);
                                 padding:3px 10px;border-radius:20px;letter-spacing:.5px'>{cfg["label"].upper()}</span>
                </div>
                <p style='font-size:14px;color:#44403c;line-height:1.8;margin:0'>{ins.get("body","")}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Summary")
    if ml:
        if task == 'classification':
            perf_sent = (f"The best classifier ({ml['best']}) achieves accuracy=<strong>{ml['accuracy']:.4f}</strong> "
                         f"and F1=<strong>{ml['f1']:.4f}</strong> predicting <em>{ml['target']}</em>.")
        else:
            r2_desc = ("strong" if ml['r2'] > 0.7 else "moderate" if ml['r2'] > 0.4 else "modest")
            perf_sent = (f"The best model ({ml['best']}) achieves R²=<strong>{ml['r2']:.4f}</strong> "
                         f"predicting <em>{ml['target']}</em> — a {r2_desc} result.")
        nc_prev = ', '.join(intake['numeric'][:4]) + ('…' if len(intake['numeric']) > 4 else '')
    else:
        perf_sent = "No models have been trained yet."
        nc_prev   = ', '.join(intake['numeric'][:4]) if intake['numeric'] else 'none'

    miss_note  = "no missing values" if intake['missing'] == 0 else f"{intake['missing']} missing values"
    comp_color = C['green'] if intake['completeness'] >= 95 else C['amber']
    st.markdown(f"""<div style='background:#fff;border:1px solid #e7e5e4;border-radius:10px;
                padding:22px 26px;font-size:15px;color:#44403c;line-height:1.9'>
        The dataset has <strong>{intake['rows']:,} records</strong> across <strong>{intake['cols']} columns</strong>,
        with <strong style='color:{comp_color}'>{intake['completeness']}% completeness</strong> ({miss_note}).
        {f'Numeric features: <em style="color:{C["blue"]}">{nc_prev}</em>.' if intake['numeric'] else ''}
        {perf_sent}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ASK QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### Ask Anything About Your Data")
    st.markdown(
        "<p style='color:#78716c;font-size:15px;margin-top:-6px;margin-bottom:18px'>"
        "Ask any question about your dataset, model performance, feature importance, "
        "correlations, or how to improve results. The agent answers from your actual data.</p>",
        unsafe_allow_html=True
    )

    if ml:
        if task == 'classification':
            examples = [
                "Which feature has highest correlation with target?",
                "Is the dataset balanced?",
                "Which model performed best?",
                f"What does F1 score of {ml['f1']:.3f} mean?",
                "What patterns exist in the data?",
                "What are the most important variables?",
            ]
        else:
            examples = [
                "Which feature has highest correlation with target?",
                "Which model performed best?",
                f"What does R² of {ml['r2']:.3f} mean?",
                "What are the most important variables?",
                "What patterns exist in the data?",
                "How can I improve the model?",
            ]
    else:
        examples = [
            "How many rows and columns does this dataset have?",
            "Which columns have missing values?",
            "Are there strong correlations between features?",
            "Which column would make the best prediction target?",
            "What data cleaning steps would you recommend?",
        ]

    # Quick example buttons
    st.markdown("<p style='font-size:13px;font-weight:600;color:#78716c;letter-spacing:.5px;"
                "text-transform:uppercase;margin-bottom:10px'>Quick example questions</p>",
                unsafe_allow_html=True)
    ex_cols = st.columns(3)
    for i, ex in enumerate(examples):
        with ex_cols[i % 3]:
            if st.button(ex, key=f"ex_{i}"):
                ctx = build_context(df, eda, intake, ml, task)
                answer = ask_ai_question(ex, ctx, df, eda, intake, ml, task)
                st.session_state.chat_history.append({'role': 'user',    'content': ex})
                st.session_state.chat_history.append({'role': 'assistant','content': answer})
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f"""<div style='background:#f7f5f2;border:1px solid #e7e5e4;border-radius:9px;
                        padding:14px 18px;margin-bottom:9px'>
                <div style='font-size:12px;font-weight:600;color:#78716c;letter-spacing:.5px;margin-bottom:5px'>YOU</div>
                <div style='font-size:15px;color:#1c1917'>{msg['content']}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style='background:#fff;border:1px solid #e7e5e4;
                        border-left:3px solid {C["teal"]};border-radius:9px;
                        padding:14px 18px;margin-bottom:9px'>
                <div style='font-size:12px;font-weight:600;color:{C["teal"]};letter-spacing:.5px;margin-bottom:5px'>DATAANALYTICA</div>
                <div style='font-size:15px;color:#44403c;line-height:1.8'>{msg['content']}</div>
            </div>""", unsafe_allow_html=True)

    # Input form
    with st.form("qa_form", clear_on_submit=True):
        question  = st.text_input(
            "Type any question about your data",
            placeholder="e.g. Which feature is most correlated with target? Is the data balanced?"
        )
        submitted = st.form_submit_button("Send →")

    if submitted and question.strip():
        ctx = build_context(df, eda, intake, ml, task)
        answer = ask_ai_question(question, ctx, df, eda, intake, ml, task)
        st.session_state.chat_history.append({'role': 'user',    'content': question})
        st.session_state.chat_history.append({'role': 'assistant','content': answer})
        st.rerun()
