import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
MODEL_PATH = "model_laptop.pkl"  # Ubah path ini sesuai dengan file model Anda

def load_model():
    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file tidak ditemukan. Pastikan file model_laptop.pkl tersedia.")
        return None

# Preprocessing Function
def preprocess_input(input_data):
    # Buat DataFrame dari data input
    df = pd.DataFrame([input_data])

    # Lakukan preprocessing manual
    df_encoded = pd.get_dummies(df, columns=["Company", "TypeName", "OpSys"], drop_first=True)

    # Dapatkan kolom yang diharapkan oleh model
    model = load_model()
    if model:
        expected_columns = model.feature_names_in_
    else:
        expected_columns = []  # Fallback jika model tidak dimuat

    # Pastikan semua kolom yang diharapkan model tersedia
    for col in expected_columns:
        if col not in df_encoded:
            df_encoded[col] = 0

    # Urutkan kolom agar sesuai model
    df_encoded = df_encoded.reindex(columns=expected_columns, fill_value=0)

    return df_encoded

# Predict Price Function
def predict_price(model, input_data):
    # Preprocess input
    df = preprocess_input(input_data)
    return model.predict(df)[0]

# Dashboard Components
st.title("Prediksi Harga Laptop")
st.write("Masukkan spesifikasi laptop untuk memprediksi harga.")

# User Inputs
company = st.selectbox("Perusahaan", ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI"])
type_name = st.selectbox("Tipe Laptop", ["Ultrabook", "Notebook", "Gaming", "Workstation", "2 in 1 Convertible"])
inches = st.number_input("Ukuran Layar (Inci)", min_value=10.0, max_value=20.0, step=0.1)
screen_resolution = st.text_input("Resolusi Layar", "1920x1080")
cpu = st.text_input("Tipe Prosesor", "Intel Core i5")
ram = st.slider("RAM (GB)", min_value=4, max_value=64, step=4)
memory = st.text_input("Penyimpanan", "256GB SSD")
gpu = st.text_input("Tipe GPU", "Intel UHD Graphics")
opsys = st.selectbox("Sistem Operasi", ["Windows", "macOS", "Linux", "No OS", "Chrome OS"])
weight = st.number_input("Berat (kg)", min_value=0.5, max_value=5.0, step=0.1)

# Predict Button
if st.button("Prediksi Harga"):
    model = load_model()
    if model:
        input_data = {
            "Company": company,
            "TypeName": type_name,
            "Inches": inches,
            "ScreenResolution": screen_resolution,
            "Cpu": cpu,
            "Ram": ram,
            "Memory": memory,
            "Gpu": gpu,
            "OpSys": opsys,
            "Weight": weight,
        }
        
        try:
            predicted_price = predict_price(model, input_data)
            st.success(f"Prediksi harga laptop: Rp {predicted_price:,.2f}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Gagal memuat model. Tidak dapat melakukan prediksi.")
