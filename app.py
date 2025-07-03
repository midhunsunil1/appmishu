
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules
from utils import load_data, get_numeric_df

st.set_page_config(page_title="FocusNest Dashboard", page_icon="🪺", layout="wide")
st.sidebar.image("assets/FocusNest_logo.png", width=180)
st.sidebar.title("FocusNest")
st.sidebar.write("Build Better Habits. Break the Social Cycle.")

DATA_PATH = "data/Dataset_for_Business_AssociationRuleReady.xlsx"
df = load_data(DATA_PATH)

tabs = st.tabs(["Visualization","Classification","Clustering","Assoc Rules","Regression"])

# Visualization
with tabs[0]:
    st.header("Data Visualization")
    sns.set_style("whitegrid")
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Age")
        fig,ax=plt.subplots()
        sns.histplot(df['Age'], ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Minutes Spent")
        fig,ax=plt.subplots()
        sns.kdeplot(df['Daily_Minutes_Spent'], ax=ax, shade=True)
        st.pyplot(fig)
    st.dataframe(df.head())

# Classification
with tabs[1]:
    st.header("Classification")
    X = get_numeric_df(df)
    y = df['Willingness_to_Subscribe'].map({'No':0,'Maybe':1,'Yes':2})
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, stratify=y, random_state=42)
    models = {"KNN":KNeighborsClassifier(),
              "DT":DecisionTreeClassifier(),
              "RF":RandomForestClassifier(),
              "GBRT":GradientBoostingClassifier()}
    results={}
    for name,model in models.items():
        model.fit(X_train,y_train)
        pred=model.predict(X_test)
        results[name]=[accuracy_score(y_test,pred),
                       precision_score(y_test,pred,average='macro'),
                       recall_score(y_test,pred,average='macro'),
                       f1_score(y_test,pred,average='macro')]
    res_df=pd.DataFrame(results,index=['Acc','Prec','Rec','F1']).T
    st.dataframe(res_df)
    sel=st.selectbox("Confusion Matrix",list(models.keys()))
    if st.button("Show CM"):
        cm=confusion_matrix(y_test,models[sel].predict(X_test))
        st.write(cm)

# Clustering
with tabs[2]:
    st.header("Clustering")
    k=st.slider("k",2,10,4)
    km=KMeans(n_clusters=k, random_state=42).fit(get_numeric_df(df))
    df['Cluster']=km.labels_
    st.write(df.groupby('Cluster').mean())
    fig,ax=plt.subplots()
    sns.scatterplot(x='Daily_Minutes_Spent',y='Monthly_Income',hue='Cluster',data=df,ax=ax)
    st.pyplot(fig)

# Association Rules
with tabs[3]:
    st.header("Association Rules")
    bin_cols=[c for c in df.columns if set(df[c].unique())<= {0,1}]
    cols=st.multiselect("Columns",bin_cols,default=bin_cols[:2])
    sup=st.slider("min_support",0.01,0.2,0.05)
    if cols:
        rules=association_rules(apriori(df[cols],min_support=sup,use_colnames=True),metric="confidence",min_threshold=0.3)
        st.dataframe(rules[['antecedents','consequents','support','confidence','lift']].head(10))

# Regression
with tabs[4]:
    st.header("Regression")
    target='Pay_Amount'
    X=get_numeric_df(df).drop(columns=[target])
    y=df[target]
    models={'Linear':LinearRegression(),'Ridge':Ridge(),'Lasso':Lasso()}
    for name,model in models.items():
        model.fit(X,y)
        st.write(f"{name} R2:", model.score(X,y))



# --- Predict Pay Amount Tab ---
with st.container():
    if 'tab_predict' in tab_dict and tab_dict['tab_predict']:
        st.header("📈 Predict Pay Amount (All Models)")
        df = load_data()
        key_inputs = ["Age", "Monthly_Income", "Daily_Minutes_Spent", "Willingness", "Main_Challenge"]
        target = "Pay_Amount"

        # Drop target, encode categoricals for training
        X_orig = df.drop(columns=[target])
        X_enc = pd.get_dummies(X_orig, drop_first=True)
        y = df[target]
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y, test_size=0.3, random_state=1)
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
                "R2": round(r2_score(y_te, y_pred), 3),
                "MSE": round(mean_squared_error(y_te, y_pred), 3),
                "MAE": round(mean_absolute_error(y_te, y_pred), 3)
            })

        st.markdown("#### Manual Input for Prediction (only 5 variables)")
        with st.form("manual_pred_form"):
            input_dict = {}
            for col in key_inputs:
                if df[col].dtype in [int, float]:
                    input_dict[col] = st.number_input(col, value=float(df[col].mean()))
                else:
                    options = sorted([str(o) for o in df[col].dropna().unique()])
                    input_dict[col] = st.selectbox(col, options, key=f"manual_{col}")
            submitted = st.form_submit_button("Predict for All Models")
        if submitted:
            # Set all other features to mean/mode
            full_row = {}
            for col in X_orig.columns:
                if col in key_inputs:
                    full_row[col] = input_dict[col]
                elif df[col].dtype in [int, float]:
                    full_row[col] = float(df[col].mean())
                else:
                    full_row[col] = df[col].mode().iloc[0]
            inp_df = pd.DataFrame([full_row])
            inp_enc = pd.get_dummies(inp_df, drop_first=True)
            inp_enc = inp_enc.reindex(columns=X_enc.columns, fill_value=0)
            preds = {}
            for name, model in reg_models.items():
                try:
                    pred = float(model.predict(inp_enc)[0])
                    preds[name] = round(pred, 2)
                except Exception:
                    preds[name] = "Err"
            perf_df = pd.DataFrame(reg_perf).set_index("Model")
            perf_df["Manual Prediction"] = pd.Series(preds)
            st.dataframe(perf_df.style.background_gradient(axis=0, cmap='Oranges'))

            # Bar chart, matching your color theme
            valid_preds = {k: v for k, v in preds.items() if isinstance(v, float) or isinstance(v, int)}
            if valid_preds:
                import plotly.graph_objects as go
                bar_colors = ["#FF9B54", "#FFC285", "#FFB26B", "#FFA447"]  # orange gradient
                fig = go.Figure([go.Bar(x=list(valid_preds.keys()), y=list(valid_preds.values()), marker_color=bar_colors[:len(valid_preds)])])
                fig.update_layout(title="Pay Amount Prediction (All Models)", yaxis_title="Predicted Pay Amount", xaxis_title="Model", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
