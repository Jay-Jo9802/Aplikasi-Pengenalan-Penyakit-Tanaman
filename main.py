import os
import threading
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import io
from PIL import Image
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.applications.mobilenet import preprocess_input
import streamlit as st
import requests
import json

# Nonaktifkan optimisasi oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app)

# Lokasi dan nama file model penyakit tanaman
disease_model_dir = "model"
disease_model_filename = "model_penyakit.h5"
disease_model_path = os.path.join(disease_model_dir, disease_model_filename)

# Variabel global untuk model penyakit
disease_model = None

# Informasi penyakit tanaman
disease_info = {
    'BACTERIAL SPOT': {
        'Deskripsi': "Penyakit yang disebabkan oleh bakteri pada daun dan buah tanaman, umumnya menyebabkan bercak berwarna gelap.",
        'Penanganan': "Pengendalian dengan menggunakan fungisida bakterisida, pengelolaan tanaman yang baik, dan praktik sanitasi."
    },
    'BLACK MEASLES': {
        'Deskripsi': "Penyakit yang disebabkan oleh jamur, terlihat sebagai bercak-bercak kecil berwarna coklat atau hitam pada daun.",
        'Penanganan': "Pengendalian dengan menyemprotkan fungisida, menjaga kebersihan lingkungan, dan menghindari kelembaban berlebih."
    },
    'BLACK ROT': {
        'Deskripsi': "Penyakit yang disebabkan oleh jamur pada tanaman cruciferous, biasanya terlihat sebagai bercak-bercak hitam pada daun.",
        'Penanganan': "Pengendalian dengan menjaga kebersihan, menggunakan bibit yang bebas penyakit, dan menghindari kondisi lembab."
    },
    'CITRUS GREENING': {
        'Deskripsi': "Penyakit serius pada tanaman jeruk yang disebabkan oleh bakteri, terlihat sebagai daun menguning dan pertumbuhan tidak normal.",
        'Penanganan': "Pengendalian dengan mengelola serangga penular vektor, pemangkasan, dan aplikasi antibiotik tertentu."
    },
    'LEAF BLIGHT': {
        'Deskripsi': "Penyakit yang disebabkan oleh jamur pada daun tanaman, terlihat sebagai bercak berwarna coklat atau abu-abu pada daun.",
        'Penanganan': "Pengendalian dengan menyemprotkan fungisida, menghilangkan daun yang terinfeksi, dan menjaga kebersihan."
    },
    'LEAF MOLD': {
        'Deskripsi': "Penyakit yang disebabkan oleh jamur pada daun, umumnya terlihat sebagai bercak putih keabu-abuan pada permukaan daun.",
        'Penanganan': "Pengendalian dengan memastikan sirkulasi udara yang baik, menghilangkan daun yang terinfeksi, dan mengelola kelembaban."
    },
    'LEAF SCORCH': {
        'Deskripsi': "Penyakit yang menyebabkan daun mengering dan menguning, sering disebabkan oleh bakteri atau jamur patogen.",
        'Penanganan': "Pengendalian dengan pengelolaan air yang baik, penyemprotan fungisida, dan memangkas tanaman."
    },
    'LEAF SPOT': {
        'Deskripsi': "Penyakit yang menyebabkan bercak berwarna gelap atau coklat pada daun, disebabkan oleh jamur atau bakteri.",
        'Penanganan': "Pengendalian dengan menjaga kebersihan, menyemprotkan fungisida, dan mengatur irigasi."
    },
    'MOSAIC VIRUS': {
        'Deskripsi': "Penyakit virus yang umum pada tanaman, terlihat sebagai daun menguning dengan pola mosaik atau bercak.",
        'Penanganan': "Pengendalian dengan menggunakan bibit bebas virus, mengelola serangga vektor, dan menghilangkan tanaman yang terinfeksi."
    },
    'POWDERY MILDEW': {
        'Deskripsi': "Penyakit jamur yang terlihat sebagai serbuk putih pada daun dan bagian tanaman lainnya.",
        'Penanganan': "Pengendalian dengan menyemprotkan fungisida, menjaga sirkulasi udara yang baik, dan menjaga tanaman tetap kering."
    },
    'RUST': {
        'Deskripsi': "Penyakit yang disebabkan oleh jamur, terlihat sebagai bercak-bercak berwarna coklat atau oranye pada daun dan batang.",
        'Penanganan': "Pengendalian dengan menggunakan bibit resisten, menyemprotkan fungisida, dan menjaga kebersihan."
    },
    'SCAB': {
        'Deskripsi': "Penyakit jamur yang menyebabkan bercak berwarna gelap atau abu-abu pada daun, buah, dan batang tanaman.",
        'Penanganan': "Pengendalian dengan menjaga kebersihan, menggunakan bibit bebas penyakit, dan menyemprotkan fungisida."
    },
    'SPIDER MITES': {
        'Deskripsi': "Hama kecil yang menyebabkan kerusakan pada tanaman dengan membuat jaringan halus di bawah daun.",
        'Penanganan': "Pengendalian dengan menyemprotkan insektisida, menjaga kelembaban udara, dan membuang daun yang terinfeksi."
    },
    'TARGET SPOT': {
        'Deskripsi': "Penyakit yang disebabkan oleh jamur pada daun tanaman, terlihat sebagai bercak berwarna gelap dengan tepi merah.",
        'Penanganan': "Pengendalian dengan menyemprotkan fungisida, menjaga sirkulasi udara yang baik, dan memangkas tanaman."
    },
    'YELLOW LEAF CURL VIRUS': {
        'Deskripsi': "Penyakit virus yang menyebabkan daun tanaman menguning, keriput, dan berkumpul.",
        'Penanganan': "Pengendalian dengan menggunakan bibit bebas virus, mengelola serangga vektor, dan menghilangkan tanaman yang terinfeksi."
    }
}

# Daftar kelas penyakit tanaman
class_names_disease = list(disease_info.keys())

def load_and_process_image(image_data, target_size=(224, 224)):
    img = Image.open(image_data)
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    global disease_model
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Data gambar tidak ditemukan'}), 400

        image_file = request.files['image']
        image_data = io.BytesIO(image_file.read())
        img_array = load_and_process_image(image_data)

        if disease_model is None:
            # Muat model penyakit jika belum dimuat
            if os.path.exists(disease_model_path):
                # custom_objects = {
                #     'MobileNet': mobilenet.MobileNet
                # }
                # Load the model
                # disease_model = tf.keras.models.load_model(disease_model_path, custom_objects=custom_objects)
                disease_model = tf.keras.models.load_model(disease_model_path)
                print(f"Model penyakit berhasil dimuat dari {disease_model_path}")
            else:
                return jsonify({'error': f'File model penyakit tidak ditemukan di {disease_model_path}'}), 500

        # Lakukan prediksi
        predictions = disease_model.predict(img_array)
        predicted_class = tf.argmax(predictions[0]).numpy()
        confidence = predictions[0][predicted_class]
        # is_above_threshold = bool(confidence > 0.5)

        # Dapatkan informasi penyakit tanaman
        predicted_disease = class_names_disease[predicted_class]
        disease_details = disease_info[predicted_disease]

        # Kembalikan hasil prediksi
        return jsonify({
            'message': 'Prediksi penyakit tanaman berhasil.',
            'data': {
                'hasil': predicted_disease,
                'skorKepercayaan': float(confidence * 100),
                # 'isAboveThreshold': is_above_threshold,
                'Deskripsi': disease_details['Deskripsi'],
                'Penanganan': disease_details['Penanganan']
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

def run_flask():
    app.run(host="0.0.0.0", port=8000)

# Jalankan Flask dalam thread terpisah
threading.Thread(target=run_flask).start()

# Code Streamlit

# Title
st.set_page_config(
    page_title="Pengenalan Penyakit Tanaman",
    page_icon="🌿",
)
# Sidebar Navigasi

st.sidebar.title("Navigasi")
option = st.sidebar.selectbox("Pilih Halaman", ["Beranda", "Tentang", "Pengenalan Penyakit"])

# Beranda 
if option == "Beranda":
    st.title("Aplikasi Pengenalan Penyakit Tanaman")
    image_path = "home1.png"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Selamat datang di Aplikasi Pengenalan Penyakit Tanaman! 🌿🔍

    Tujuan dibuatnya aplikasi ini untuk membantu mengidentifikasi penyakit tanaman secara efisien. Unggah gambar tanaman, dan aplikasi akan menganalisisnya untuk mendeteksi tanda-tanda penyakit. Bersama-sama, mari lindungi tanaman kita dan pastikan panen yang lebih sehat!

    ### Cara Penggunaan
    1. **Unggah Gambar:** Buka Halaman **Pengenalan Penyakit** dan pilih gambar tanaman yang dicurigai terkena penyakit.
    2. **Prediksi:** Aplikasi akan memproses gambar menggunakan algoritma canggih untuk mengidentifikasi potensi penyakit.
    3. **Hasil:** Lihat hasil dan rekomendasi untuk tindakan lebih lanjut.

    ### Mulai Sekarang
    Klik pada halaman **Pengenalan Penyakit** di sidebar untuk mengunggah gambar dan rasakan kekuatan Sistem Pengenalan Penyakit Tanaman!

    ### Tentang Pengenalan Penyakit Tanaman 
    Pelajari lebih lanjut tentang project penulis di halaman **Tentang**.
    """)

# Tentang 
elif option == "Tentang":
    st.title("Tentang")
    st.write("Aplikasi ini menggunakan model pembelajaran mendalam untuk mengidentifikasi berbagai penyakit tanaman dari gambar. Model ini telah dilatih pada dataset gambar tanaman dan dapat memprediksi penyakit dengan akurasi yang tinggi.")
    st.markdown("""
    ### Mengapa Memilih Pengenalan Penyakit Tanaman?
    - **Akurasi:** Sistem kami menggunakan teknik pembelajaran mesin mutakhir untuk deteksi penyakit yang akurat.
    - **Mudah Digunakan:** Antarmuka yang sederhana dan intuitif untuk pengalaman pengguna yang lancar.
    - **Cepat dan Efisien:** Dapatkan hasil dalam hitungan detik, memungkinkan pengambilan keputusan yang cepat.
                
    #### Dataset
    Dataset ini berupa gambar 15 jenis penyakit tanaman dapat dilihat pada Google Drive Penulis, [Klik untuk melihat](https://drive.google.com/drive/folders/1--Y32Rv_80vrnv6c4VKH_0rLQeeBl9PV). 
    Untuk melatih model dalam mengenali dan mengklasifikasikan penyakit tanaman, penulis menggunakan dataset yang diambil dari Kaggle dan GitHub Plant Village, [Kaggle Plant Village](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) dan [Github Plant Village](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color).
    Proyek Penulisan Ilmiah ini dapat dilihat di GitHub penulis. [Klik di sini untuk mengakses repositori](https://github.com/Jay-Jo9802/Aplikasi-Pengenalan-Penyakit-Tanaman).

    Dataset ini terdiri dari 4500 gambar daun tanaman yang terkena penyakit yang dikategorikan ke dalam 15 kelas yang berbeda,
    total dataset dibagi menjadi rasio 70/20 untuk set pelatihan dan validasi untuk menjaga struktur direktori. 
    #### Content
    1. train (3150 gambar)
    2. test (450 gambar)
    3. validation (900 gambar)

    ### Kelas Penyakit
    1. Tanaman Apel
        - Apple Black Rot
        - Apple Rust
        - Apple Scab
    2. Tanaman Cherry
        - Cherry Powdery
    3. Tanaman Jagung
        - Corn Leaf Blight
        - Corn Leaf Spot
    4. Tanaman Anggur
        - Grape Black Measles
    5. Tanaman Jeruk
        - Orange Citrus Greening
    6. Tanaman Persik
        - Peach Bacterial Spot
    7. Tanaman Strawberry
        - Strawberry Leaf Scorch
    8. Tanaman Tomat
        - Tomato Leaf Mold
        - Tomato Mosaic Virus
        - Tomato Spider Mites
        - Tomato Target Spot
        - Tomato Yellow Leaf Curl Virus

    ### Tentang Penulis
    Anda dapat terhubung dengan penulis melalui media sosial berikut:
    - **LinkedIn**: [Johan](https://www.linkedin.com/in/johan-jayjo/)
    - **GitHub**: [Jay-Jo9802](https://github.com/Jay-Jo9802)
    - **Instagram**: [@jay_jo9802](https://www.instagram.com/jay_jo9802/)

    Jangan ragu untuk menghubungi penulis jika Anda memiliki pertanyaan, saran, atau ingin berkolaborasi dalam proyek yang menarik!

    """)

# Pengenalan Penyakit 
elif option == "Pengenalan Penyakit":
    st.title("Pengenalan Penyakit Tanaman")
    st.write("Unggah gambar daun tanaman untuk memprediksi penyakit.")

    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Inisialisasi status sesi untuk menampilkan/menyembunyikan gambar
        if "show_image" not in st.session_state:
            st.session_state.show_image = True

        # Reset hasil prediksi saat gambar baru diunggah
        st.session_state.prediction_result = None

        # Fungsi untuk melakukan prediksi menggunakan Flask API
        def predict_disease(image_data):
            FLASK_API_URL = "http://localhost:8000/predict/disease"
            try:
                # Simpan BytesIO ke dalam file sementara untuk dikirim
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(image_data.read())
                temp_file.close()

                # Load file sementara dan kirim ke Flask API
                with open(temp_file.name, 'rb') as f:
                    files = {'image': f}
                    response = requests.post(FLASK_API_URL, files=files)

                # Hapus file sementara setelah penggunaan
                os.remove(temp_file.name)

                if response.status_code == 200:
                    return response.json()
                else:
                    return {'error': f'Error: {response.text}'}
            except Exception as e:
                return {'error': f'Error: {str(e)}'}

        # Button untuk show/hide gambar
        if st.button("Tampilkan/Sembunyikan Gambar"):
            st.session_state.show_image = not st.session_state.show_image

        # Fungsi untuk menampilkan gambar jika show_image bernilai True
        if st.session_state.show_image and uploaded_file is not None:
            # Menampilkan gambar tanpa membuka Image baru
            st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)

        # Button untuk melakukan prediksi
        if st.button("Prediksi"):
            if uploaded_file is None:
                st.warning("Tolong unggah gambar terlebih dahulu.")
            else:
                # Reset hasil prediksi sebelum membuat prediksi baru
                st.session_state.prediction_result = None

                # Kirim gambar untuk diprediksi
                with st.spinner('Memprediksi...'):
                    img_byte_arr = io.BytesIO(uploaded_file.read())
                    prediction_result = predict_disease(img_byte_arr)

                if 'error' in prediction_result:
                    st.error(prediction_result['error'])
                else:
                    st.session_state.prediction_result = prediction_result
                    st.success("Prediksi Berhasil!")

        # Menampilkan hasil prediksi jika tersedia
        if st.session_state.prediction_result:
            prediction_result = st.session_state.prediction_result
            st.write("**Penyakit Terdeteksi:**", prediction_result['data']['hasil'])
            st.write("**Akurasi Keyakinan:**", f"{prediction_result['data']['skorKepercayaan']:.2f}%") 
            st.write("**Deskripsi:**", prediction_result['data']['Deskripsi'])
            st.write("**Cara Penanganan:**", prediction_result['data']['Penanganan'])
