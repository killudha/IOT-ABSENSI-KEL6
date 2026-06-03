import cv2
import os
import requests
import json
import time

# --- KONFIGURASI FIREBASE ---
# Ganti dengan URL Realtime Database Anda
FIREBASE_URL = "https://aiot-absensi-kel6-default-rtdb.firebaseio.com/mahasiswa.json"

print("=== SISTEM REGISTRASI WAJAH KELOMPOK 6 ===")
nim = input("Masukkan NIM: ")
nama = input("Masukkan Nama Lengkap: ")

# 1. Simpan Data Teks ke Firebase
data_mahasiswa = {nim: {"nama": nama, "nim": nim}}
requests.patch(FIREBASE_URL, data=json.dumps(data_mahasiswa))
print("Data teks berhasil disimpan ke Firebase!")

# 2. Persiapan Folder Dataset
dataset_dir = f"dataset/{nama}"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# 3. Mulai Pengambilan Foto (Webcam)
cap = cv2.VideoCapture(0)
count = 0
max_fotos = 30 # Jumlah foto yang akan diambil (bisa dinaikkan jika mau)

print("\n[INFO] Menginisialisasi kamera...")
print("[INSTRUKSI] Tatap kamera, lalu gerakkan wajah perlahan ke kiri, kanan, atas, dan bawah.")
time.sleep(2)

while count < max_fotos:
    ret, frame = cap.read()
    if not ret: break

    # Menampilkan instruksi di layar
    cv2.putText(frame, f"Mengambil data: {count}/{max_fotos}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Gerakkan kepala perlahan", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Registrasi Wajah - KEL6", frame)

    # Ambil foto setiap kali tombol 'c' (Capture) ditekan, 
    # ATAU otomatis pakai timer (di sini kita pakai otomatis tiap 0.2 detik agar cepat)
    waktu_sekarang = time.time()
    
    # Otomatis jepret dan simpan gambar
    file_name = f"{dataset_dir}/{count}.jpg"
    cv2.imwrite(file_name, frame)
    count += 1
    
    time.sleep(0.2) # Jeda antar foto agar angle sempat berubah
    
    if cv2.waitKey(1) == 27: # Tekan ESC untuk batal
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n[SUKSES] Berhasil mengambil {count} foto untuk {nama}.")
print("Registrasi Selesai!")
