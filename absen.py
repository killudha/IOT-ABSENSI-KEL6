# --- LOAD DATASET WAJAH (MULTI-ANGLE) ---
known_face_encodings = []
known_face_names = []

dataset_dir = "dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("[INFO] Memproses dataset wajah. Ini mungkin memakan waktu beberapa detik...")

# Loop ke dalam setiap folder mahasiswa
for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    
    if os.path.isdir(person_dir):
        # Loop semua foto di dalam folder mahasiswa tersebut
        for file in os.listdir(person_dir):
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(person_dir, file)
                
                # Baca dan encode wajah
                img = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(img)
                
                # Jika di foto tersebut ditemukan wajah, simpan encodings-nya
                if len(encodings) > 0:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(person_name) # Namanya tetap sama untuk semua 30 foto

print(f"[SUKSES] Total {len(known_face_names)} data wajah (dari berbagai angle) berhasil dimuat!")