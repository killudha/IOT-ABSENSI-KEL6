from flask import Flask, render_template, Response, request, jsonify
import cv2
import face_recognition
import paho.mqtt.client as mqtt
import requests
import json
import os
import time
from datetime import datetime
from threading import Lock, Thread
import numpy as np

app = Flask(__name__)

# --- KONFIGURASI FIREBASE ---
BROKER = "broker.hivemq.com"
TOPIC = "iot/absensi/kel6"
FIREBASE_URL = "https://aiot-absensi-kel6-default-rtdb.asia-southeast1.firebasedatabase.app"

try:
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
except AttributeError:
    mqtt_client = mqtt.Client()
mqtt_client.connect(BROKER, 1883, 60)
mqtt_client.loop_start()

# --- GLOBAL VARIABLES ---
known_face_encodings, known_face_names, known_face_classes = [], [], []
dataset_dir = "dataset"
mahasiswa_sudah_absen = set()

kamera_mode = "idle" 
current_attendance_class, reg_nama, reg_kelas = "", "", ""
reg_count = 0
MAX_FOTO = 30

is_camera_on = False 
camera = None
camera_lock = Lock()
is_reloading_ai = False 

def load_dataset_wajah():
    global known_face_encodings, known_face_names, known_face_classes
    known_face_encodings.clear()
    known_face_names.clear()
    known_face_classes.clear()
    if not os.path.exists(dataset_dir): os.makedirs(dataset_dir)
    for kelas_name in os.listdir(dataset_dir):
        kelas_dir = os.path.join(dataset_dir, kelas_name)
        if os.path.isdir(kelas_dir):
            for person_name in os.listdir(kelas_dir):
                person_dir = os.path.join(kelas_dir, person_name)
                if os.path.isdir(person_dir):
                    for file in os.listdir(person_dir):
                        if file.endswith((".jpg", ".png")):
                            img_path = os.path.join(person_dir, file)
                            img = face_recognition.load_image_file(img_path)
                            encodings = face_recognition.face_encodings(img)
                            if encodings:
                                known_face_encodings.append(encodings[0])
                                known_face_names.append(person_name)
                                known_face_classes.append(kelas_name)
    print(f"[SUKSES] Dataset dimuat: {len(known_face_names)} data wajah.")

load_dataset_wajah()

def kirim_data_background(name, kelas, waktu_format):
    try:
        mqtt_client.publish(TOPIC, "ABSEN_OK")
        data_absen = {"nama": name, "kelas": kelas, "waktu": waktu_format}
        requests.post(f"{FIREBASE_URL}/log_absensi.json", data=json.dumps(data_absen), timeout=2)
    except Exception as e:
        print("Error MQTT/Firebase:", e)

def reload_ai_background():
    global is_reloading_ai
    load_dataset_wajah()
    is_reloading_ai = False

def generate_frames():
    global kamera_mode, reg_nama, reg_kelas, reg_count, is_camera_on, camera, current_attendance_class, mahasiswa_sudah_absen, is_reloading_ai
    last_ai_check_time, last_capture_time = 0, 0 
    cached_face_locations, cached_face_names = [], []
    
    while True:
        if not is_camera_on:
            blank_frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_frame, "KAMERA TIDAK AKTIF", (130, 240), cv2.FONT_HERSHEY_DUPLEX, 1, (200, 200, 200), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5) 
            continue
            
        with camera_lock:
            if camera is None: continue
            success, frame = camera.read()
            
        if not success: 
            time.sleep(0.1)
            continue
            
        current_time = time.time()
        if is_reloading_ai:
            cv2.putText(frame, "AI SINKRONISASI DATA BARU...", (20, 80), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 165, 255), 2)
            
        if kamera_mode == "idle":
            cv2.putText(frame, "SIAP DIGUNAKAN", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
        
        elif kamera_mode == "register":
            if current_time - last_capture_time > 0.15:
                person_dir = os.path.join(dataset_dir, reg_kelas, reg_nama)
                os.makedirs(person_dir, exist_ok=True)
                file_name = f"{person_dir}/{reg_count}.jpg"
                cv2.imwrite(file_name, frame)
                reg_count += 1
                last_capture_time = current_time
            
            cv2.rectangle(frame, (20, 20), (350, 80), (255, 200, 0), cv2.FILLED)
            cv2.putText(frame, f"Foto: {reg_count} / {MAX_FOTO}", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
            
            if reg_count >= MAX_FOTO:
                kamera_mode = "idle"
                is_reloading_ai = True
                Thread(target=reload_ai_background).start() 
                
        elif kamera_mode == "attendance":
            active_encodings = [enc for enc, cls in zip(known_face_encodings, known_face_classes) if cls == current_attendance_class]
            active_names = [name for name, cls in zip(known_face_names, known_face_classes) if cls == current_attendance_class]

            if not is_reloading_ai and current_time - last_ai_check_time > 0.4:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                cached_face_locations = face_recognition.face_locations(rgb_small_frame)
                cached_face_names = []
                
                if len(active_encodings) > 0 and len(cached_face_locations) > 0:
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, cached_face_locations)
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(active_encodings, face_encoding)
                        name = "Tidak Dikenal"
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = active_names[first_match_index]
                            if name not in mahasiswa_sudah_absen:
                                mahasiswa_sudah_absen.add(name) 
                                waktu_format = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                Thread(target=kirim_data_background, args=(name, current_attendance_class, waktu_format)).start()
                        cached_face_names.append(name)
                else:
                    cached_face_names = ["Tidak Ada Data"] * len(cached_face_locations)
                last_ai_check_time = current_time

            for (top, right, bottom, left), name in zip(cached_face_locations, cached_face_names):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                warna = (0, 0, 255) if name in ["Tidak Dikenal", "Tidak Ada Data"] else (0, 255, 0)
                teks_layar = f"{name} (Oke)" if name in mahasiswa_sudah_absen else name
                cv2.rectangle(frame, (left, top), (right, bottom), warna, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), warna, cv2.FILLED)
                cv2.putText(frame, teks_layar, (left + 6, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.rectangle(frame, (10, 10), (350, 70), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, f"Kelas : {current_attendance_class}", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)
            cv2.putText(frame, f"Hadir : {len(mahasiswa_sudah_absen)} Orang", (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- ROUTES FLASK ---
@app.route('/')
def dashboard(): return render_template('dashboard.html', active_page='dashboard')

@app.route('/registrasi')
def registrasi(): return render_template('registrasi.html', active_page='registrasi')

@app.route('/master')
def master(): return render_template('master.html', active_page='master')

@app.route('/history')
def history(): return render_template('history.html', active_page='history')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/kamera', methods=['POST'])
def toggle_kamera():
    global is_camera_on, camera, kamera_mode
    status = request.json.get('status')
    with camera_lock:
        if status == 'on' and not is_camera_on:
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Instan di Windows!
            is_camera_on = True
            kamera_mode = "idle"
            return jsonify({"status": "sukses", "pesan": "Kamera menyala."})
        elif status == 'off' and is_camera_on:
            kamera_mode = "idle"
            if camera is not None: camera.release(); camera = None
            is_camera_on = False
            return jsonify({"status": "sukses", "pesan": "Kamera dimatikan."})
    return jsonify({"status": "error", "pesan": "Invalid"})

@app.route('/api/attendance', methods=['POST'])
def toggle_attendance():
    global kamera_mode, current_attendance_class, is_camera_on, mahasiswa_sudah_absen
    if not is_camera_on: return jsonify({"status": "error", "pesan": "Kamera off!"})
    action, kelas = request.json.get('action'), request.json.get('kelas')
    if action == 'start' and kelas:
        current_attendance_class = kelas
        kamera_mode = "attendance"
        mahasiswa_sudah_absen.clear() 
        return jsonify({"status": "sukses"})
    elif action == 'stop':
        kamera_mode = "idle"
        current_attendance_class = ""
        return jsonify({"status": "sukses"})
    return jsonify({"status": "error"})

def registrasi_background(kelas, nim, nama):
    try:
        data_mahasiswa = {nim: {"nama": nama, "nim": nim, "kelas": kelas}}
        requests.patch(f"{FIREBASE_URL}/mahasiswa/{kelas}.json", data=json.dumps(data_mahasiswa), timeout=3)
    except Exception as e:
        print("Gagal registrasi DB:", e)

@app.route('/api/mulai_registrasi', methods=['POST'])
def mulai_registrasi():
    global kamera_mode, reg_nama, reg_kelas, reg_count, is_camera_on, camera
    data = request.json
    reg_nama, reg_kelas, reg_count, kamera_mode = data.get('nama'), data.get('kelas'), 0, "register"
    
    def setup_hardware():
        global is_camera_on, camera
        with camera_lock:
            if not is_camera_on:
                camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                is_camera_on = True
        registrasi_background(reg_kelas, data.get('nim'), reg_nama)

    Thread(target=setup_hardware).start()
    return jsonify({"status": "sukses"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)