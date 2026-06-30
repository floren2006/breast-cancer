# CDSS Klasifikasi Kanker Payudara — Streamlit App

Aplikasi Streamlit untuk mendukung skripsi klasifikasi kanker payudara
(Logistic Regression, Breast Cancer Wisconsin Dataset).

## Struktur Folder

```
streamlit_app/
├── app.py                 # Aplikasi utama (semua halaman)
├── build_artifacts.py     # Script untuk melatih ulang model & membuat artefak
├── requirements.txt
├── README.md
└── model/
    ├── data.csv               # Dataset (format kompatibel dengan notebook asli)
    ├── best_model_logreg.pkl  # Model terbaik (pipeline scaler + Logistic Regression)
    ├── benchmark_stats.pkl    # Statistik mean/std per kelas (training set)
    └── results.pkl            # Semua hasil tuning, kurva ROC/PR, confusion matrix, dll.
```

## Cara Menjalankan

1. Buat virtual environment (opsional tapi disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

4. Buka browser ke `http://localhost:8501`.

## Halaman Aplikasi

| Halaman | Deskripsi |
|---|---|
| 🏠 Dashboard | Ringkasan penelitian dan performa model |
| 📊 Dataset Explorer | Eksplorasi data, statistik deskriptif, missing value, korelasi |
| 📈 Insight Analytics | 5 insight: Feature Importance, PR Curve, Confusion Matrix, Feature Distribution, Hyperparameter Convergence |
| 🧬 Patient Prediction | Input data pasien baru → prediksi Benign/Malignant |
| 🔍 Prediction Explanation | Penjelasan kontribusi fitur terhadap prediksi terakhir |
| 📋 Model Evaluation | ROC Curve, PR Curve, Confusion Matrix, Classification Report per metode tuning |
| ℹ️ About Model | Penjelasan dataset, algoritma, alur penelitian, glosarium fitur medis |

## Catatan Penting Mengenai Dataset

Karena dataset asli (`data.csv` yang dipakai di notebook, sumber Kaggle/UCI)
tidak disertakan dalam upload, `model/data.csv` pada paket ini dibangun ulang
dari dataset bawaan scikit-learn (`sklearn.datasets.load_breast_cancer`),
yang merupakan sumber data yang **sama persis** (Breast Cancer Wisconsin
Diagnostic, UCI), hanya diformat ulang agar nama kolom & strukturnya identik
dengan notebook (`radius_mean`, `texture_mean`, ..., `concave points_mean`,
dst). Jika Anda memiliki file `data.csv` asli dari notebook, Anda dapat
menggantinya di folder `model/` lalu menjalankan ulang `build_artifacts.py`
untuk melatih ulang model dengan data tersebut — strukturnya 100% kompatibel
karena nama dan urutan kolom identik dengan notebook asli.

## Melatih Ulang Model

Jika ingin mengganti dataset atau mengubah strategi tuning, edit dan jalankan:
```bash
python build_artifacts.py
```
Script ini akan menjalankan ulang seluruh pipeline (Default, GridSearchCV,
RandomizedSearchCV, Optuna) dan menyimpan kembali seluruh file di `model/`.

## Troubleshooting

- **ModuleNotFoundError** → pastikan `pip install -r requirements.txt` sudah dijalankan di environment yang aktif.
- **FileNotFoundError pada model/** → pastikan folder `model/` tetap berada satu folder dengan `app.py` (jangan dipindah terpisah).
- **Versi scikit-learn berbeda** → jika muncul warning saat load `.pkl`, jalankan ulang `build_artifacts.py` di environment Anda agar model dilatih ulang dengan versi library yang sama.
