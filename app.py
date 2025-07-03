
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="📈 Productivity App Market Dashboard", layout="wide")
st.title("📈 Productivity App Market Analysis Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv('sample_data.csv')
    return df

df = load_data()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=[object, "category", "bool"]).columns.tolist()

tab_vis, tab_clf, tab_clu, tab_arm, tab_reg = st.tabs(
    ["📊 Data Visualization", "🤖 Classification", "👥 Clustering", "🔗 Assoc Rules", "📈 Regression"]
)

with tab_vis:
    st.header("Quick Viz")
    if numeric_cols:
        col = st.selectbox("Histogram column:", numeric_cols)
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig, use_container_width=True)

with tab_reg:
    st.header("📈 Regression: Manual Input & All Model Results")
    if not numeric_cols:
        st.warning("No numeric columns found.")
    else:
        target = st.selectbox("Select numeric target:", numeric_cols, key="reg_target", index=numeric_cols.index("Pay_Amount") if "Pay_Amount" in numeric_cols else 0)
        X_orig = df.drop(columns=[target])
        # Encode categoricals
        X_enc = pd.get_dummies(X_orig, drop_first=True)
        y = df[target]
        if len(df) < 10:
            st.warning("Not enough rows for meaningful regression modeling.")
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y, test_size=0.3, random_state=1)
            reg_models = {
                "Linear": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "DecisionTree": DecisionTreeRegressor(random_state=1)
            }
            reg_perf = []
            for name, model in reg_models.items():
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                reg_perf.append({
                    "Model": name,
                    "R2": np.round(r2_score(y_te, y_pred), 3),
                    "MSE": np.round(mean_squared_error(y_te, y_pred), 3),
                    "MAE": np.round(mean_absolute_error(y_te, y_pred), 3)
                })

            st.subheader("Manual Input for Prediction")
            input_dict = {}
            with st.form("manual_pred_form"):
                for col in X_orig.columns:
                    if str(X_orig[col].dtype).startswith("float") or str(X_orig[col].dtype).startswith("int"):
                        val = float(df[col].mean()) if np.issubdtype(X_orig[col].dtype, np.number) else 0
                        input_dict[col] = st.number_input(col, value=val, format="%.2f")
                    else:
                        # Remove NaN and force to str for all
                        options = sorted([str(o) for o in df[col].dropna().unique()])
                        input_dict[col] = st.selectbox(col, options, key=f"manual_{col}")
                submitted = st.form_submit_button("Predict for All Models")
            if submitted:
                inp_df = pd.DataFrame([input_dict])
                inp_enc = pd.get_dummies(inp_df, drop_first=True)
                inp_enc = inp_enc.reindex(columns=X_enc.columns, fill_value=0)
                preds = {}
                for name, model in reg_models.items():
                    try:
                        pred = float(model.predict(inp_enc)[0])
                        preds[name] = round(pred, 2)
                    except Exception as e:
                        preds[name] = "Err"
                perf_df = pd.DataFrame(reg_perf).set_index("Model")
                perf_df["Manual Prediction"] = pd.Series(preds)
                st.dataframe(perf_df)
                valid_preds = {k: v for k, v in preds.items() if isinstance(v, float) or isinstance(v, int)}
                if valid_preds:
                    chart = pd.DataFrame(valid_preds, index=["Prediction"]).T
                    st.bar_chart(chart)
