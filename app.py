from flask import Flask, render_template, Response, request, jsonify, session, redirect, url_for
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
app.secret_key = 'rahasia_aiot_kelompok6'

# --- KONFIGURASI FIREBASE & MQTT ---
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
current_attendance_class = ""
reg_nama, reg_kelas = "", ""
reg_count = 0
MAX_FOTO = 30
capture_trigger = False 

is_camera_on = False 
camera = None
camera_lock = Lock()
is_reloading_ai = False 

ai_thread_active = False
cached_face_locations = []
cached_face_names = []

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

def kirim_data_background(name, kelas, waktu_format, timestamp_ms):
    try:
        mqtt_client.publish(TOPIC, "ABSEN_OK")
        data_absen = {"nama": name, "kelas": kelas, "waktu": waktu_format, "timestamp": timestamp_ms}
        requests.post(f"{FIREBASE_URL}/log_absensi.json", data=json.dumps(data_absen), timeout=3)
    except Exception as e:
        print("Error Firebase/MQTT:", e)

def process_ai_thread(rgb_small_frame, active_encodings, active_names, target_class):
    global cached_face_locations, cached_face_names, ai_thread_active, mahasiswa_sudah_absen
    try:
        locations = face_recognition.face_locations(rgb_small_frame)
        names = []
        if len(active_encodings) > 0 and len(locations) > 0:
            encodings = face_recognition.face_encodings(rgb_small_frame, locations)
            for face_encoding in encodings:
                matches = face_recognition.compare_faces(active_encodings, face_encoding)
                name = "Tidak Dikenal"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = active_names[first_match_index]
                    if name not in mahasiswa_sudah_absen:
                        mahasiswa_sudah_absen.add(name) 
                        waktu_format = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        timestamp_ms = int(time.time() * 1000) # Untuk sinkronisasi UI Frontend Realtime
                        Thread(target=kirim_data_background, args=(name, target_class, waktu_format, timestamp_ms)).start()
                names.append(name)
        else:
            names = ["Tidak Ada Data"] * len(locations)
        
        cached_face_locations = locations
        cached_face_names = names
    except Exception as e:
        print("AI Error:", e)
    finally:
        ai_thread_active = False

def reload_ai_background():
    global is_reloading_ai
    load_dataset_wajah()
    is_reloading_ai = False

def generate_frames():
    global kamera_mode, reg_nama, reg_kelas, reg_count, is_camera_on, camera, current_attendance_class
    global mahasiswa_sudah_absen, is_reloading_ai, capture_trigger, ai_thread_active
    global cached_face_locations, cached_face_names
    
    last_ai_check_time = 0 
    
    while True:
        if not is_camera_on:
            blank_frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_frame, "KAMERA TIDAK AKTIF", (130, 240), cv2.FONT_HERSHEY_DUPLEX, 1, (200, 200, 200), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5) 
            continue
            
        with camera_lock:
            if camera is None or not camera.isOpened():
                time.sleep(0.1)
                continue
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
            cv2.rectangle(frame, (20, 20), (350, 80), (255, 200, 0), cv2.FILLED)
            cv2.putText(frame, f"Frame: {reg_count} / {MAX_FOTO}", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
            
            if capture_trigger:
                person_dir = os.path.join(dataset_dir, reg_kelas, reg_nama)
                os.makedirs(person_dir, exist_ok=True)
                file_name = f"{person_dir}/{reg_count}.jpg"
                cv2.imwrite(file_name, frame)
                reg_count += 1
                capture_trigger = False 
                frame = np.zeros((480, 640, 3), np.uint8) 
                
                if reg_count >= MAX_FOTO:
                    kamera_mode = "idle"
                    is_reloading_ai = True
                    Thread(target=reload_ai_background).start() 
                
        elif kamera_mode == "attendance":
            active_encodings = [enc for enc, cls in zip(known_face_encodings, known_face_classes) if cls == current_attendance_class]
            active_names = [name for name, cls in zip(known_face_names, known_face_classes) if cls == current_attendance_class]

            if not is_reloading_ai and not ai_thread_active and (current_time - last_ai_check_time > 0.3):
                ai_thread_active = True
                last_ai_check_time = current_time
                # Skala gambar diperkecil agar sangat ringan di CPU
                small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                Thread(target=process_ai_thread, args=(rgb_small_frame, active_encodings, active_names, current_attendance_class)).start()

            for (top, right, bottom, left), name in zip(cached_face_locations, cached_face_names):
                top *= 5; right *= 5; bottom *= 5; left *= 5 # Skala balik x5 karena resize fx=0.2
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

# --- CEK SESI LOGIN ---
def login_required(f):
    def wrap(*args, **kwargs):
        if 'logged_in' not in session: return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

# --- ROUTES MVC FLASK ---
@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username')
        pwd = request.form.get('password')
        if user == 'admin' and pwd == '12345678':
            session['logged_in'] = True
            session['role'] = 'admin'
            return redirect(url_for('dashboard'))
        elif user == 'absensi' and pwd == '12345678':
            session['logged_in'] = True
            session['role'] = 'operator'
            return redirect(url_for('absensi_kelas'))
        return render_template('Login.html', error="Username atau Password salah!")
    return render_template('Login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard(): 
    if session.get('role') != 'admin': return redirect(url_for('absensi_kelas'))
    return render_template('Dashboard.html', active_page='dashboard', user_role=session.get('role'))

@app.route('/absensi-kelas')
@login_required
def absensi_kelas(): 
    return render_template('AbsensiKelas.html', active_page='absensi', user_role=session.get('role'))

@app.route('/daftar-wajah')
@login_required
def daftar_wajah(): 
    return render_template('DaftarWajah.html', active_page='registrasi', user_role=session.get('role'))

@app.route('/manajemen-kelas')
@login_required
def manajemen_kelas(): 
    if session.get('role') != 'admin': return redirect(url_for('absensi_kelas'))
    return render_template('ManajemenKelas.html', active_page='master', user_role=session.get('role'))

@app.route('/detail-kelas/<kelas_id>')
@login_required
def detail_kelas(kelas_id): 
    if session.get('role') != 'admin': return redirect(url_for('absensi_kelas'))
    return render_template('DetailKelas.html', active_page='master', kelas_id=kelas_id, user_role=session.get('role'))

@app.route('/riwayat-absensi')
@login_required
def riwayat_absensi(): 
    if session.get('role') != 'admin': return redirect(url_for('absensi_kelas'))
    return render_template('RiwayatAbsensi.html', active_page='history', user_role=session.get('role'))

@app.route('/video_feed')
def video_feed(): 
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- API CONTROL KAMERA & AI ---
@app.route('/api/kamera_off', methods=['POST'])
def auto_kamera_off():
    global is_camera_on, camera, kamera_mode
    with camera_lock:
        kamera_mode = "idle"
        if camera is not None: 
            camera.release()
            camera = None
        is_camera_on = False
    return "OK", 200

@app.route('/api/kamera', methods=['POST'])
def toggle_kamera():
    global is_camera_on, camera, kamera_mode
    status = request.json.get('status')
    with camera_lock:
        if status == 'on' and not is_camera_on:
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            # OPTIMASI ANTI LAG: Batasi resolusi hardware langsung dari sumbernya
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            is_camera_on = True
            kamera_mode = "idle"
            return jsonify({"status": "sukses"})
        elif status == 'off' and is_camera_on:
            kamera_mode = "idle"
            if camera is not None: 
                camera.release()
                camera = None
            is_camera_on = False
            return jsonify({"status": "sukses"})
    return jsonify({"status": "error"})

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

def simpan_db_background(kelas, nim, nama, role):
    try:
        data_user = {"nama": nama, "id": nim, "kelas": kelas, "role": role}
        requests.put(f"{FIREBASE_URL}/users/{kelas}/{nim}.json", data=json.dumps(data_user), timeout=5)
    except Exception as e:
        pass

@app.route('/api/mulai_registrasi', methods=['POST'])
def mulai_registrasi():
    global kamera_mode, reg_nama, reg_kelas, reg_count, is_camera_on, camera
    data = request.json
    reg_nama, reg_kelas, role, nim = data.get('nama'), data.get('kelas'), data.get('role'), data.get('nim')
    reg_count = 0
    kamera_mode = "register" 
    
    def setup_hardware():
        global is_camera_on, camera
        with camera_lock:
            if not is_camera_on:
                camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                is_camera_on = True
        simpan_db_background(reg_kelas, nim, reg_nama, role)

    Thread(target=setup_hardware).start()
    return jsonify({"status": "sukses"})

@app.route('/api/capture', methods=['POST'])
def manual_capture():
    global capture_trigger, reg_count
    if kamera_mode == "register":
        capture_trigger = True
        time.sleep(0.2) 
        return jsonify({"status": "sukses", "count": reg_count})
    return jsonify({"status": "error"})

if __name__ == '__main__':
    # Threaded=True memastikan web tidak ngefreeze saat AI memproses
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)