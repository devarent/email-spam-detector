"""
# ==============================================================================
# TUGAS: PERCOBAAN MENDETEKSI EMAIL SPAM
# Mata Kuliah : Keamanan Berbasis Sistem Cerdas
# Dosen       : HERIANTO, S.Pd., M.T.
# Kelas       : TIFM
# Nama        : Muhammad Arya Fatthurahman
# NIM         : 2023230006
# Referensi   : 1. Modul Spam Mail Prediction using ML with Python (TF-IDF & Logistic Regression)
#               2. "AI dalam Cyber Security" (Onno W. Purbo) - Bab Algoritma ML & Studi Kasus Spam
# ==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    precision_score, 
    recall_score, 
    f1_score
)

def main():
    print("=" * 90)
    print("PROGRAM PENDETEKSI EMAIL SPAM & ANALISIS ALGORITMA CYBERSECURITY")
    print("Mahasiswa : Muhammad Arya Fatthurahman | NIM: 2023230006 | Kelas: TIFM")
    print("=" * 90)

    # 1. Load Data
    print("\n[1] Membaca dataset 'mail_data.csv'...")
    raw_mail_data = pd.read_csv('mail_data.csv')
    print(f"    Dimensi data: {raw_mail_data.shape[0]} baris x {raw_mail_data.shape[1]} kolom")

    # 2. Penanganan Missing Value
    mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

    # 3. Label Encoding (spam = 0, ham = 1)
    mail_data['Category'] = mail_data['Category'].astype(object)
    mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

    X = mail_data['Message']
    Y = mail_data['Category'].astype(int)

    total = len(Y)
    spam_n = (Y == 0).sum()
    ham_n = (Y == 1).sum()
    print(f"    Distribusi: Ham = {ham_n:,} ({ham_n/total*100:.1f}%), Spam = {spam_n:,} ({spam_n/total*100:.1f}%) [Data Imbalance]")

    # 4. Pembagian Data Latih dan Uji (80:20)
    print("\n[2] Membagi data latih dan uji (80:20, random_state=3)...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=3
    )
    print(f"    Data latih : {X_train.shape[0]} sampel")
    print(f"    Data uji   : {X_test.shape[0]} sampel")

    # 5. Ekstraksi Fitur TF-IDF
    print("\n[3] Ekstraksi fitur teks menggunakan TF-IDF Vectorizer...")
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)
    print(f"    Jumlah fitur kata unik (vocabulary): {len(feature_extraction.get_feature_names_out()):,}")

    # 6. Pelatihan Tiga Algoritma (Logistic Regression, Naive Bayes, Random Forest)
    print("\n[4] Melatih Model Machine Learning:")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=3),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=3)
    }

    predictions = {}
    for name, model_obj in models.items():
        model_obj.fit(X_train_features, Y_train)
        predictions[name] = model_obj.predict(X_test_features)
        print(f"    ✓ Model [{name}] berhasil dilatih!")

    # 7. Evaluasi Model Komparatif
    print("\n[5] TABEL PERBANDINGAN PERFORMA ALGORITMA:")
    print("-" * 90)
    print(f"{'Model':<26} | {'Akurasi Latih':<13} | {'Akurasi Uji':<11} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 90)

    for name, model_obj in models.items():
        p_train = model_obj.predict(X_train_features)
        p_test = predictions[name]
        acc_tr = accuracy_score(Y_train, p_train) * 100
        acc_ts = accuracy_score(Y_test, p_test) * 100
        prec = precision_score(Y_test, p_test) * 100
        rec = recall_score(Y_test, p_test) * 100
        f1 = f1_score(Y_test, p_test) * 100
        print(f"{name:<26} | {acc_tr:>12.2f}% | {acc_ts:>10.2f}% | {prec:>8.2f}% | {rec:>7.2f}% | {f1:>7.2f}%")
    print("-" * 90)

    # 8. Inferensi Contoh Baru (Termasuk Obfuscation Attack dari Onno W. Purbo)
    print("\n[6] Uji Coba Inferensi Pesan Baru (Termasuk Obfuscation Attack):")
    contoh_email = [
        "Congratulations! You have won a $1,000 Walmart Gift Card. Click here to claim your prize immediately: http://bit.ly/claim-prize",
        "Hi Arya, remember we have a team meeting scheduled tomorrow at 10 AM to discuss our intelligent security systems project. See you there!",
        "URGENT: Your account has been suspended! Verify your banking credentials within 24 hours to restore access or call 0800123456.",
        "Selamat siang Pak Herianto, mohon izin mengumpulkan tugas praktikum percobaan deteksi email spam.",
        "FR33 entry in a weekly comp to W1N cash prize! Cl!ck to claim your reward now!"
    ]
    test_feat = feature_extraction.transform(contoh_email)

    for i, msg in enumerate(contoh_email, 1):
        print(f"\n    [{i}] Pesan: \"{msg}\"")
        for name, model_obj in models.items():
            pred = model_obj.predict(test_feat[i-1:i])[0]
            status = 'HAM (Bukan Spam)' if pred == 1 else 'SPAM MAIL'
            if hasattr(model_obj, 'predict_proba'):
                conf = model_obj.predict_proba(test_feat[i-1:i])[0][pred] * 100
                print(f"        - {name:<24}: {status} ({conf:.1f}% confidence)")
            else:
                print(f"        - {name:<24}: {status}")

    print("\n" + "=" * 90)
    print("Program selesai dijalankan.")
    print("=" * 90)

if __name__ == '__main__':
    main()
