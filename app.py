from flask import Flask, render_template, Response
import cv2
import sys
import os

# Paksa Python mencari model di lokasi yang benar
import face_recognition_models
model_path = os.path.dirname(face_recognition_models.pose_predictor_model_location())
sys.path.append(model_path)

from flask import Flask, render_template, Response
import face_recognition
# ... (lanjutkan import lainnya seperti cv2, mqtt, dll)
import paho.mqtt.client as mqtt
import requests
import json
import os
import time
from datetime import datetime

app = Flask(__name__)

# --- KONFIGURASI ---
BROKER = "broker.hivemq.com"
TOPIC = "iot/absensi/kel6"
FIREBASE_URL = "https://aiot-absensi-kel6-default-rtdb.firebaseio.com/log_absensi.json"

# Inisialisasi MQTT
mqtt_client = mqtt.Client()
mqtt_client.connect(BROKER, 1883, 60)
mqtt_client.loop_start()

# --- LOAD DATASET WAJAH ---
known_face_encodings = []
known_face_names = []
dataset_dir = "dataset"

print("[INFO] Memuat model wajah...")
if os.path.exists(dataset_dir):
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if os.path.isdir(person_dir):
            for file in os.listdir(person_dir):
                if file.endswith((".jpg", ".png")):
                    image_path = os.path.join(person_dir, file)
                    img = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(person_name)
print(f"[SUKSES] {len(known_face_names)} data wajah dimuat.")

terakhir_absen = {}
camera = cv2.VideoCapture(0)

def generate_frames():
    """Fungsi generator untuk memproses frame dan mengirimkannya ke web"""
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Proses Face Recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                
                # Logika Absensi (Cooldown 60 detik)
                waktu_skrg = time.time()
                if name not in terakhir_absen or (waktu_skrg - terakhir_absen[name] > 60):
                    terakhir_absen[name] = waktu_skrg
                    print(f"Absen Berhasil: {name}")
                    
                    # Kirim MQTT ke ESP32 di Wokwi
                    mqtt_client.publish(TOPIC, "ABSEN_OK")
                    
                    # Simpan ke Firebase
                    waktu_format = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    data_absen = {"nama": name, "waktu": waktu_format}
                    requests.post(FIREBASE_URL, data=json.dumps(data_absen))

            # Gambar kotak hijau/merah di wajah
            top, right, bottom, left = face_location
            top *= 4; right *= 4; bottom *= 4; left *= 4
            warna = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), warna, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warna, 2)

        # Ubah frame menjadi format JPEG untuk Web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- ROUTES WEB ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Jalankan server di http://127.0.0.1:5000/
    app.run(debug=True, host='0.0.0.0', port=5000)