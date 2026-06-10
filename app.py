import streamlit as st
import pickle

# Load Model dan Vectorizer
model = pickle.load(open('model_nlp.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Deteksi Sentimen Ulasan Tokopedia 🛒")
st.write("Ketik ulasan sebuah produk di bawah ini, dan AI akan menebak apakah ini ulasan yang Positif atau Negatif!")

# Membuat kotak input teks untuk pengguna
ulasan_user = st.text_area("Masukkan Ulasan Produk (Contoh: Barang bagus banget, pengiriman cepat!):")

# Tombol untuk memproses
if st.button("Analisis Sentimen"):
    if ulasan_user:
        # 1. Terjemahkan teks input pengguna jadi angka pakai vectorizer
        teks_vector = vectorizer.transform([ulasan_user])
        
        # 2. Lakukan Prediksi
        prediksi = model.predict(teks_vector)[0]
        
        # 3. Tampilkan Hasil dengan tampilan menarik
        if prediksi == 'Positif':
            st.success(f"Hasil Analisis: **{prediksi}** 😃")
        else:
            st.error(f"Hasil Analisis: **{prediksi}** 😡")
    else:
        st.warning("Silakan ketik ulasan terlebih dahulu sebelum dianalisis.")
