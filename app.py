"""
Clinical Decision Support System (CDSS) — Klasifikasi Kanker Payudara
Logistic Regression | Breast Cancer Wisconsin (UCI)
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="CDSS — Klasifikasi Kanker Payudara",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"


# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "best_model_logreg.pkl")


@st.cache_resource
def load_benchmark():
    return joblib.load(MODEL_DIR / "benchmark_stats.pkl")


@st.cache_resource
def load_results():
    return joblib.load(MODEL_DIR / "results.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv(MODEL_DIR / "data.csv")
    df = df.drop(columns=["Unnamed: 32"], errors="ignore")
    df["diagnosis_label"] = df["diagnosis"].map({"M": "Malignant", "B": "Benign"})
    return df


try:
    model = load_model()
    benchmark = load_benchmark()
    results = load_results()
    df_raw = load_data()
except FileNotFoundError as e:
    st.error(
        "File model/data tidak ditemukan. Pastikan folder `model/` "
        "(berisi best_model_logreg.pkl, benchmark_stats.pkl, results.pkl, data.csv) "
        "berada di direktori yang sama dengan app.py."
    )
    st.stop()

FEATURE_NAMES = benchmark["feature_names"]
mean_benign = benchmark["mean_benign"]
mean_malignant = benchmark["mean_malignant"]

best_name = results["best_name"]
best_metrics = results["best_metrics"]
all_results = results["all_results"]
coef_df = results["coef_df"]

PRIMARY = "#6c5ce7"
BENIGN_COLOR = "#2ecc71"
MALIGNANT_COLOR = "#e74c3c"

# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("🩺 CDSS Kanker Payudara")
st.sidebar.caption("Logistic Regression · Breast Cancer Wisconsin")

PAGES = [
    "🏠 Dashboard",
    "📊 Dataset Explorer",
    "📈 Insight Analytics",
    "🧬 Patient Prediction",
    "🔍 Prediction Explanation",
    "📋 Model Evaluation",
    "ℹ️ About Model",
]
page = st.sidebar.radio("Navigasi", PAGES, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption(f"Model terbaik saat ini: **{best_name}**")
st.sidebar.caption(f"Recall: **{best_metrics['Recall']*100:.2f}%**")

# Session state to carry a prediction from "Patient Prediction" page
# to "Prediction Explanation" page.
if "last_patient_input" not in st.session_state:
    st.session_state.last_patient_input = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


def metric_card_row(metrics_dict):
    cols = st.columns(len(metrics_dict))
    for col, (k, v) in zip(cols, metrics_dict.items()):
        col.metric(k, f"{v*100:.2f}%")


# ====================================================================
# PAGE 1 — DASHBOARD
# ====================================================================
if page == "🏠 Dashboard":
    st.title("🏠 Dashboard Penelitian")
    st.markdown(
        "**Klasifikasi Kanker Payudara (Benign vs Malignant)** menggunakan "
        "**Logistic Regression** pada dataset *Breast Cancer Wisconsin (Diagnostic)*."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sampel", f"{len(df_raw)}")
    c2.metric("Total Fitur", f"{len(FEATURE_NAMES)}")
    c3.metric("Jumlah Kelas", "2 (Benign / Malignant)")

    st.divider()
    st.subheader(f"Performa Model Terbaik — {best_name}")
    metric_card_row(best_metrics)

    st.divider()
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Distribusi Kelas Diagnosis")
        vc = df_raw["diagnosis_label"].value_counts()
        fig = px.pie(
            names=vc.index, values=vc.values,
            color=vc.index,
            color_discrete_map={"Benign": BENIGN_COLOR, "Malignant": MALIGNANT_COLOR},
            hole=0.45,
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.subheader("Ringkasan Metodologi")
        st.markdown(
            """
            - **Pipeline anti data-leakage**: split → scaling (fit hanya di train) → model
            - **Tuning**: GridSearchCV, RandomizedSearchCV, Optuna (Bayesian)
            - **Pemilihan model**: prioritas **Recall → F1 → ROC-AUC**
            - **Alasan**: meminimalkan *False Negative* (pasien ganas yang lolos terdeteksi)
              merupakan risiko klinis paling kritis
            """
        )
        st.info(
            f"Model terpilih: **{best_name}**, dengan Recall "
            f"**{best_metrics['Recall']*100:.2f}%** pada test set.",
            icon="✅",
        )

    st.divider()
    st.caption(
        "Gunakan menu di sidebar untuk menjelajahi dataset, melihat insight penelitian, "
        "atau melakukan prediksi pasien baru."
    )

# ====================================================================
# PAGE 2 — DATASET EXPLORER
# ====================================================================
elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.caption("Breast Cancer Wisconsin (Diagnostic) Dataset — UCI Machine Learning Repository")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Data & Filter", "Statistik Deskriptif", "Missing Value", "Korelasi Fitur"]
    )

    with tab1:
        st.subheader("Filter Data")
        col1, col2 = st.columns([1, 3])
        with col1:
            diag_filter = st.multiselect(
                "Filter diagnosis", options=["Benign", "Malignant"],
                default=["Benign", "Malignant"],
            )
            n_rows = st.slider("Jumlah baris ditampilkan", 5, 100, 15)
        filtered = df_raw[df_raw["diagnosis_label"].isin(diag_filter)]
        with col2:
            st.dataframe(filtered.head(n_rows), use_container_width=True)
        st.caption(f"Menampilkan {min(n_rows, len(filtered))} dari {len(filtered)} baris terfilter.")

    with tab2:
        st.subheader("Statistik Deskriptif")
        numeric_cols = df_raw.select_dtypes(include=np.number).columns.drop("id", errors="ignore")
        st.dataframe(df_raw[numeric_cols].describe().T.round(3), use_container_width=True)

    with tab3:
        st.subheader("Missing Value Check")
        miss = df_raw.isna().sum()
        miss = miss[miss.index != "diagnosis_label"]
        miss_df = pd.DataFrame({"Kolom": miss.index, "Jumlah Missing": miss.values})
        total_missing = miss_df["Jumlah Missing"].sum()
        if total_missing == 0:
            st.success("Tidak ditemukan missing value pada dataset (selain kolom artefak yang sudah dibuang).", icon="✅")
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Korelasi Antar Fitur (Heatmap)")
        corr_features = st.multiselect(
            "Pilih fitur untuk korelasi (kosongkan = semua fitur _mean)",
            options=FEATURE_NAMES,
            default=[f for f in FEATURE_NAMES if f.endswith("_mean")][:10],
        )
        if corr_features:
            corr = df_raw[corr_features].corr()
            fig = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                aspect="auto", zmin=-1, zmax=1,
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Pilih minimal satu fitur untuk menampilkan heatmap.")

# ====================================================================
# PAGE 3 — INSIGHT ANALYTICS
# ====================================================================
elif page == "📈 Insight Analytics":
    st.title("📈 Insight Analytics")
    st.caption("Lima insight utama dari hasil penelitian, ditampilkan secara interaktif.")

    insight_tabs = st.tabs([
        "1️⃣ Feature Importance", "2️⃣ PR Curve & Probability",
        "3️⃣ Confusion Matrix", "4️⃣ Feature Distribution", "5️⃣ Hyperparameter Convergence",
    ])

    # ---------------- Insight 1 ----------------
    with insight_tabs[0]:
        st.subheader("Feature Importance — Koefisien & Odds Ratio")
        top_n = st.radio("Tampilkan", [5, 10, 15], horizontal=True, index=1)
        top_df = coef_df.head(top_n)

        fig = px.bar(
            top_df.sort_values("Koefisien"), x="Koefisien", y="Fitur", orientation="h",
            color="Koefisien", color_continuous_scale="RdYlGn_r",
            title=f"Top {top_n} Fitur Pemicu Malignant (Koefisien Logistic Regression)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_df, use_container_width=True, hide_index=True)
        most_imp = coef_df.iloc[0]["Fitur"]
        st.info(
            f"Fitur paling berpengaruh terhadap prediksi *Malignant* adalah **{most_imp}**. "
            "Odds Ratio > 1 menunjukkan peningkatan fitur tersebut berasosiasi dengan "
            "peningkatan kemungkinan *Malignant*.",
            icon="💡",
        )

    # ---------------- Insight 2 ----------------
    with insight_tabs[1]:
        st.subheader("Precision–Recall Curve & Probability Distribution")
        pr = results["pr_curve"]
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pr["recalls"], y=pr["precisions"], mode="lines", fill="tozeroy",
                name=f'PR Curve (AP={pr["avg_precision"]:.4f})', line=dict(color=PRIMARY, width=3),
            ))
            fig.update_layout(
                title="Precision–Recall Curve", xaxis_title="Recall", yaxis_title="Precision",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            y_test = results["y_test"]
            y_prob = results["y_pred_prob"]
            dist_df = pd.DataFrame({
                "Probabilitas": y_prob,
                "Kelas Aktual": np.where(y_test == 1, "Malignant", "Benign"),
            })
            fig2 = px.histogram(
                dist_df, x="Probabilitas", color="Kelas Aktual", nbins=30, barmode="overlay",
                color_discrete_map={"Benign": BENIGN_COLOR, "Malignant": MALIGNANT_COLOR},
                title="Distribusi Probabilitas Prediksi",
            )
            fig2.add_vline(x=0.5, line_dash="dash", line_color="black", annotation_text="Threshold 0.5")
            st.plotly_chart(fig2, use_container_width=True)
        st.success(
            f"Average Precision (AP) = **{pr['avg_precision']:.4f}**. Model memiliki probabilitas "
            "prediksi yang terpisah dengan baik sehingga mampu membedakan Benign dan Malignant.",
            icon="📈",
        )

    # ---------------- Insight 3 ----------------
    with insight_tabs[2]:
        st.subheader("Confusion Matrix Analysis")
        cm = results["confusion_matrix"]
        tn, fp, fn, tp = cm.ravel()

        col1, col2 = st.columns([1.2, 1])
        with col1:
            fig = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                x=["Benign", "Malignant"], y=["Benign", "Malignant"],
            )
            fig.update_layout(title=f"Confusion Matrix — {best_name}")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            fnr = fn / (fn + tp)
            fpr_rate = fp / (fp + tn)
            st.metric("Sensitivity (Recall)", f"{sensitivity*100:.2f}%")
            st.metric("Specificity", f"{specificity*100:.2f}%")
            st.metric("False Negative Rate", f"{fnr*100:.2f}%")
            st.metric("False Positive Rate", f"{fpr_rate*100:.2f}%")

        st.warning(
            f"Dari {tp+fn} pasien malignant sesungguhnya, model berhasil mendeteksi {tp} pasien "
            f"({sensitivity*100:.2f}%) dan hanya melewatkan {fn} pasien (FN). "
            "Recall tinggi penting karena meminimalkan risiko pasien ganas yang tidak terdeteksi.",
            icon="⚠️",
        )

    # ---------------- Insight 4 ----------------
    with insight_tabs[3]:
        st.subheader("Feature Distribution Analysis")
        selected_feature = st.selectbox("Pilih fitur", options=FEATURE_NAMES, index=FEATURE_NAMES.index("texture_mean") if "texture_mean" in FEATURE_NAMES else 0)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df_raw, x=selected_feature, color="diagnosis_label", barmode="overlay", nbins=30,
                color_discrete_map={"Benign": BENIGN_COLOR, "Malignant": MALIGNANT_COLOR},
                title=f"Histogram — {selected_feature}",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.box(
                df_raw, x="diagnosis_label", y=selected_feature, color="diagnosis_label",
                color_discrete_map={"Benign": BENIGN_COLOR, "Malignant": MALIGNANT_COLOR},
                title=f"Boxplot — {selected_feature}",
            )
            st.plotly_chart(fig2, use_container_width=True)

        mean_b = df_raw[df_raw["diagnosis_label"] == "Benign"][selected_feature].mean()
        mean_m = df_raw[df_raw["diagnosis_label"] == "Malignant"][selected_feature].mean()
        median_b = df_raw[df_raw["diagnosis_label"] == "Benign"][selected_feature].median()
        median_m = df_raw[df_raw["diagnosis_label"] == "Malignant"][selected_feature].median()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean Benign", f"{mean_b:.3f}")
        m2.metric("Mean Malignant", f"{mean_m:.3f}")
        m3.metric("Median Benign", f"{median_b:.3f}")
        m4.metric("Median Malignant", f"{median_m:.3f}")

        direction = "lebih tinggi" if mean_m > mean_b else "lebih rendah"
        st.info(
            f"Rata-rata **{selected_feature}** pada kelas Malignant ({mean_m:.3f}) {direction} "
            f"dibandingkan Benign ({mean_b:.3f}). Pola ini konsisten dengan kontribusi fitur "
            "terhadap koefisien model pada tab Feature Importance.",
            icon="🔬",
        )

    # ---------------- Insight 5 ----------------
    with insight_tabs[4]:
        st.subheader("Hyperparameter Convergence")
        metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        plot_df = all_results.melt(id_vars="Metode", value_vars=metric_cols, var_name="Metrik", value_name="Score")
        plot_df["Score (%)"] = plot_df["Score"] * 100

        fig = px.bar(
            plot_df, x="Metrik", y="Score (%)", color="Metode", barmode="group",
            title="Perbandingan Metrik — Default vs GridSearchCV vs RandomizedSearchCV vs Optuna",
        )
        fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Threshold 95%")
        fig.update_yaxes(range=[85, 102])
        st.plotly_chart(fig, use_container_width=True)

        display_results = all_results.copy()
        for c in metric_cols:
            display_results[c] = display_results[c].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(display_results, use_container_width=True, hide_index=True)

        st.subheader("Parameter Terbaik per Metode")
        param_tab1, param_tab2, param_tab3 = st.tabs(["GridSearchCV", "RandomizedSearchCV", "Optuna"])
        with param_tab1:
            st.json({k.replace("clf__", ""): v for k, v in results["gs_best_params"].items()})
        with param_tab2:
            st.json({k.replace("clf__", ""): v for k, v in results["rs_best_params"].items()})
        with param_tab3:
            st.json(results["opt_best_params"])

        st.success(
            "Semua metode tuning konvergen pada performa tinggi (>95%), menunjukkan model "
            "bersifat *robust* dan *reproducible* terhadap variasi metode pencarian hyperparameter.",
            icon="✅",
        )

# ====================================================================
# PAGE 4 — PATIENT PREDICTION
# ====================================================================
elif page == "🧬 Patient Prediction":
    st.title("🧬 Patient Prediction")
    st.caption("Masukkan nilai fitur hasil pemeriksaan sel untuk memprediksi diagnosis.")

    with st.expander("ℹ️ Cara penggunaan", expanded=False):
        st.markdown(
            "Isi nilai fitur di bawah ini (default = nilai rata-rata dataset), lalu klik "
            "**Predict**. Anda juga bisa memuat *sample pasien* acak dari dataset sebagai titik awal."
        )

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("🎲 Muat Sampel Pasien Acak"):
            sample = df_raw.sample(1, random_state=np.random.randint(0, 10000)).iloc[0]
            st.session_state.sample_loaded = {f: float(sample[f]) for f in FEATURE_NAMES}
            st.session_state.sample_true_label = sample["diagnosis_label"]
    with col_btn2:
        if st.button("↩️ Reset ke Rata-rata Dataset"):
            st.session_state.sample_loaded = None
            st.session_state.sample_true_label = None

    if "sample_loaded" not in st.session_state:
        st.session_state.sample_loaded = None
        st.session_state.sample_true_label = None

    if st.session_state.sample_loaded:
        st.caption(f"Sampel dimuat (label asli dataset: **{st.session_state.sample_true_label}**) — silakan edit nilai jika perlu.")

    st.divider()
    st.subheader("Input Fitur Pasien")

    feature_groups = {
        "Mean": [f for f in FEATURE_NAMES if f.endswith("_mean")],
        "Standard Error (SE)": [f for f in FEATURE_NAMES if f.endswith("_se")],
        "Worst": [f for f in FEATURE_NAMES if f.endswith("_worst")],
    }

    patient_input = {}
    group_tabs = st.tabs(list(feature_groups.keys()))
    for tab, (group_name, feats) in zip(group_tabs, feature_groups.items()):
        with tab:
            cols = st.columns(3)
            for i, feat in enumerate(feats):
                default_val = (
                    st.session_state.sample_loaded[feat]
                    if st.session_state.sample_loaded
                    else float(df_raw[feat].mean())
                )
                with cols[i % 3]:
                    patient_input[feat] = st.number_input(
                        feat, value=round(default_val, 4), format="%.4f", key=f"input_{feat}",
                    )

    st.divider()
    predict_clicked = st.button("🔮 Predict", type="primary", use_container_width=True)

    if predict_clicked:
        X_new = pd.DataFrame([patient_input])[FEATURE_NAMES]
        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0]
        label = "Malignant" if pred == 1 else "Benign"
        confidence = prob[pred]

        st.session_state.last_patient_input = patient_input
        st.session_state.last_prediction = {
            "label": label, "prob_malignant": prob[1], "prob_benign": prob[0],
        }

        st.divider()
        if label == "Malignant":
            st.error(f"### 🩺 Prediksi: **{label}**", icon="🚨")
        else:
            st.success(f"### 🩺 Prediksi: **{label}**", icon="✅")

        c1, c2, c3 = st.columns(3)
        c1.metric("Diagnosis Prediksi", label)
        c2.metric("Probabilitas Malignant", f"{prob[1]*100:.2f}%")
        c3.metric("Probabilitas Benign", f"{prob[0]*100:.2f}%")

        fig = go.Figure(go.Bar(
            x=[prob[0]*100, prob[1]*100], y=["Benign", "Malignant"], orientation="h",
            marker_color=[BENIGN_COLOR, MALIGNANT_COLOR], text=[f"{prob[0]*100:.1f}%", f"{prob[1]*100:.1f}%"],
            textposition="auto",
        ))
        fig.update_layout(title="Probabilitas Prediksi", xaxis_title="Probabilitas (%)", height=250)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Buka menu **🔍 Prediction Explanation** di sidebar untuk melihat faktor "
            "apa saja yang paling memengaruhi hasil prediksi ini.",
            icon="👉",
        )

# ====================================================================
# PAGE 5 — PREDICTION EXPLANATION
# ====================================================================
elif page == "🔍 Prediction Explanation":
    st.title("🔍 Prediction Explanation")

    if st.session_state.last_prediction is None:
        st.warning(
            "Belum ada hasil prediksi. Silakan buka menu **🧬 Patient Prediction**, "
            "isi data pasien, lalu klik Predict terlebih dahulu.",
            icon="⚠️",
        )
    else:
        pred_info = st.session_state.last_prediction
        patient_input = st.session_state.last_patient_input
        label = pred_info["label"]

        st.subheader(f"Penjelasan untuk Prediksi: {label}")
        c1, c2 = st.columns(2)
        c1.metric("Diagnosis Prediksi", label)
        c2.metric(
            "Confidence",
            f"{(pred_info['prob_malignant'] if label=='Malignant' else pred_info['prob_benign'])*100:.2f}%",
        )

        st.divider()
        st.subheader("Top Contributing Features")

        coef = model.named_steps["clf"].coef_[0]
        scaler = model.named_steps["scaler"]
        x_vec = np.array([patient_input[f] for f in FEATURE_NAMES])
        x_scaled = scaler.transform(x_vec.reshape(1, -1))[0]
        contribution = coef * x_scaled  # contribution to log-odds

        contrib_df = pd.DataFrame({
            "Fitur": FEATURE_NAMES,
            "Nilai Pasien": x_vec,
            "Kontribusi (log-odds)": contribution,
        })
        contrib_df["Arah"] = np.where(contrib_df["Kontribusi (log-odds)"] > 0, "→ Malignant", "→ Benign")
        contrib_df["Abs Kontribusi"] = contrib_df["Kontribusi (log-odds)"].abs()
        contrib_df = contrib_df.sort_values("Abs Kontribusi", ascending=False).reset_index(drop=True)

        top_k = st.slider("Jumlah fitur ditampilkan", 5, 15, 8)
        top_contrib = contrib_df.head(top_k)

        fig = px.bar(
            top_contrib.sort_values("Kontribusi (log-odds)"),
            x="Kontribusi (log-odds)", y="Fitur", orientation="h", color="Arah",
            color_discrete_map={"→ Malignant": MALIGNANT_COLOR, "→ Benign": BENIGN_COLOR},
            title="Kontribusi Fitur terhadap Prediksi (Top Contributing Features)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Perbandingan Nilai Pasien vs Rata-rata Kelas")

        compare_feats = top_contrib["Fitur"].tolist()
        compare_df = pd.DataFrame({
            "Feature": compare_feats,
            "Pasien": [patient_input[f] for f in compare_feats],
            "Mean Benign": [mean_benign[f] for f in compare_feats],
            "Mean Malignant": [mean_malignant[f] for f in compare_feats],
        }).round(3)
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Pasien", x=compare_df["Feature"], y=compare_df["Pasien"], marker_color=PRIMARY))
        fig2.add_trace(go.Bar(name="Mean Benign", x=compare_df["Feature"], y=compare_df["Mean Benign"], marker_color=BENIGN_COLOR))
        fig2.add_trace(go.Bar(name="Mean Malignant", x=compare_df["Feature"], y=compare_df["Mean Malignant"], marker_color=MALIGNANT_COLOR))
        fig2.update_layout(barmode="group", title="Nilai Pasien vs Rata-rata Benign/Malignant", xaxis_tickangle=-40)
        st.plotly_chart(fig2, use_container_width=True)

        top_feat = top_contrib.iloc[0]
        st.info(
            f"Fitur dengan kontribusi terbesar terhadap prediksi ini adalah **{top_feat['Fitur']}** "
            f"(nilai pasien = {top_feat['Nilai Pasien']:.3f}), yang mendorong prediksi ke arah "
            f"**{top_feat['Arah'].replace('→ ', '')}**.",
            icon="🔎",
        )

# ====================================================================
# PAGE 6 — MODEL EVALUATION
# ====================================================================
elif page == "📋 Model Evaluation":
    st.title("📋 Model Evaluation")

    method_choice = st.selectbox(
        "Pilih metode untuk dievaluasi", ["Default", "GridSearchCV", "RandomizedSearchCV", "Optuna"], index=1,
    )

    roc_data = results["roc_data"]
    metrics_map = {
        "Default": results["baseline_metrics"],
        "GridSearchCV": results["gs_metrics"],
        "RandomizedSearchCV": results["rs_metrics"],
        "Optuna": results["opt_metrics"],
    }
    sel_metrics = metrics_map[method_choice]

    st.subheader(f"Metrik — {method_choice}")
    metric_card_row(sel_metrics)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve (Semua Metode)")
        fig = go.Figure()
        colors_map = {"Default": "#95a5a6", "GridSearchCV": "#f39c12", "RandomizedSearchCV": "#e67e22", "Optuna": "#27ae60"}
        for name, d in roc_data.items():
            fig.add_trace(go.Scatter(
                x=d["fpr"], y=d["tpr"], mode="lines", name=f'{name} (AUC={d["auc"]:.4f})',
                line=dict(color=colors_map.get(name, PRIMARY), width=3 if name == method_choice else 1.5),
            ))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Precision–Recall Curve (Model Terbaik)")
        pr = results["pr_curve"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=pr["recalls"], y=pr["precisions"], mode="lines", fill="tozeroy", line=dict(color=PRIMARY, width=3)))
        fig2.update_layout(xaxis_title="Recall", yaxis_title="Precision", title=f"AP = {pr['avg_precision']:.4f}")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Confusion Matrix (Model Terbaik)")
    cm = results["confusion_matrix"]
    fig3 = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
        x=["Benign", "Malignant"], y=["Benign", "Malignant"],
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Classification Report (Model Terbaik)")
    from sklearn.metrics import classification_report
    report = classification_report(
        results["y_test"], results["y_pred"], target_names=["Benign", "Malignant"], output_dict=True,
    )
    report_df = pd.DataFrame(report).T.round(4)
    st.dataframe(report_df, use_container_width=True)

# ====================================================================
# PAGE 7 — ABOUT MODEL
# ====================================================================
elif page == "ℹ️ About Model":
    st.title("ℹ️ About Model")

    st.subheader("Dataset")
    st.markdown(
        """
        **Breast Cancer Wisconsin (Diagnostic) Dataset** — UCI Machine Learning Repository.
        Berisi hasil pengukuran citra digital dari *fine needle aspirate* (FNA) massa payudara,
        mendeskripsikan karakteristik inti sel yang muncul pada citra.
        - 569 sampel pasien
        - 30 fitur numerik (mean, standard error, worst dari 10 karakteristik sel)
        - Target: Diagnosis (Benign / Malignant)
        """
    )

    st.subheader("Algoritma: Logistic Regression")
    st.markdown(
        """
        Logistic Regression dipilih karena sifatnya yang **interpretable** (koefisien dapat
        diartikan sebagai log-odds), cocok untuk konteks klinis di mana penjelasan keputusan
        model menjadi pertimbangan penting, selain performa prediksi itu sendiri.
        """
    )

    st.subheader("Alur Penelitian (Workflow)")
    st.markdown(
        """
        1. **Load & EDA** — eksplorasi distribusi kelas dan fitur
        2. **Preprocessing anti data-leakage** — split data sebelum scaling, `fit` scaler
           hanya pada training set
        3. **Baseline model** — Logistic Regression default
        4. **Hyperparameter tuning** — GridSearchCV, RandomizedSearchCV, Optuna (Bayesian/TPE)
        5. **Pemilihan model terbaik** — prioritas metrik **Recall → F1 Score → ROC-AUC**
        6. **Evaluasi final** — Confusion Matrix, ROC Curve, Precision-Recall Curve
        7. **Lima insight** — Feature Importance, PR & Probability, Confusion Matrix,
           Feature Distribution, Hyperparameter Convergence
        """
    )

    st.subheader("Penjelasan Fitur Medis (Glosarium Singkat)")
    glossary = {
        "radius": "Rata-rata jarak dari pusat ke titik pada keliling inti sel",
        "texture": "Standar deviasi nilai grayscale (ukuran kekasaran tekstur sel)",
        "perimeter": "Keliling inti sel",
        "area": "Luas area inti sel",
        "smoothness": "Variasi lokal panjang radius (kehalusan kontur sel)",
        "compactness": "perimeter² / area − 1.0 (kepadatan bentuk sel)",
        "concavity": "Tingkat keparahan bagian cekung pada kontur sel",
        "concave points": "Jumlah bagian cekung pada kontur sel",
        "symmetry": "Simetri bentuk inti sel",
        "fractal dimension": "Aproksimasi 'kekasaran garis pantai' (kompleksitas kontur)",
    }
    gloss_df = pd.DataFrame({"Fitur Dasar": glossary.keys(), "Keterangan": glossary.values()})
    st.dataframe(gloss_df, use_container_width=True, hide_index=True)
    st.caption("Setiap fitur dasar di atas memiliki 3 varian dalam dataset: `_mean`, `_se` (standard error), dan `_worst`.")

    st.subheader("Referensi")
    st.markdown(
        """
        - Wolberg, W.H., Street, W.N., & Mangasarian, O.L. — *Breast Cancer Wisconsin
          (Diagnostic) Data Set*, UCI Machine Learning Repository.
        - Dokumentasi scikit-learn: Logistic Regression, Pipeline, model selection.
        """
    )

st.sidebar.divider()
st.sidebar.caption("Dibangun dengan Streamlit · Clinical Decision Support System Prototype")
