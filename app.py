import pandas as pd
import streamlit as st

# from joblib import load

import time

st.set_page_config(page_title='Prediksi Harga Properti', page_icon='🏠')

st.title('🏠 Prediksi Harga Properti')

# Read Data utama
properti = pd.read_csv('data_input/properti_jual.csv')

# Read model yang telah dibuat
model = pd.read_pickle('harga_properti_model_38.pkl')
# model = load('harga_properti_model.joblib')

# Fungsi untuk melakukan prediksi
def prediksi_harga_properti(data_input):
    data = pd.DataFrame(data_input, index=[0])
    data = pd.get_dummies(data, columns=['Sertifikat', 'Tipe.Properti', 'Kota'])

    # Menambahkan kolom yang hilang dengan nilai nol
    missing_cols = [col for col in model.feature_names_in_.tolist() if col not in data.columns]

    for col in missing_cols:
        data[col] = 0

    # Pastikan urutan kolom sesuai dengan urutan saat pelatihan
    data = data[model.feature_names_in_.tolist()]

    prediksi = model.predict(data)
    return prediksi[0]

st.sidebar.markdown('---')

## Streamlit Layout
with st.container():
    st.subheader('Masukkan spesifikasi rumah:')

    # input data
    ## Luas bangunan
    l_bangunan = st.select_slider(
        'Luas Bangunan',
        options=list(range(10,1001,10)),
        value=120
    )

    ## Kamar tidur
    k_tidur = st.number_input(
        'Jumlah kamar tidur',
        min_value = 1,
        max_value = 10,
        value = 2

    )

    ## Kamar mandi
    k_mandi = st.number_input(
        'Jumlah kamar mandi',
        min_value = 1,
        max_value = 10,
        value = 1
    )

    ## Sertifikat
    sertifikat = st.selectbox(
        'Jenis Sertifikat',
        options = properti['Sertifikat'].unique(),
    )

    ## Tipe Properti
    tipe_properti = st.selectbox(
        'Tipe Properti',
        options = properti['Tipe.Properti'].unique()
    )

    ## Kota
    kota = st.selectbox(
        'Kota',
        options = properti['Kota'].unique()
    )

    # Tombol untuk memicu prediksi
    
    if st.button('Hitung Harga Prediksi'):
        input_data = {
            'K.Mandi': k_mandi,
            'K.Tidur': k_tidur,
            'L.Bangunan': l_bangunan,
            'Sertifikat': sertifikat,
            'Tipe.Properti': tipe_properti,
            'Kota': kota
        }

        with st.spinner('Menghitung...'):
            harga = prediksi_harga_properti(input_data)
            time.sleep(2)
            
            st.success(f'Hasil Prediksi Harga Rumah: Rp {harga:,.0f}')


