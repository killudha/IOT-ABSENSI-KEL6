import cv2
import paho.mqtt.client as mqtt


BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "absensi/kelas/kel6"
PAYLOAD_ABSEN = "ABSEN_OK"


def create_mqtt_client():
    client = mqtt.Client()
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    return client


def main():
    client = create_mqtt_client()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera tidak dapat dibuka.")
        client.loop_stop()
        client.disconnect()
        return

    print("IOT-ABSENSI-KEL6 Aktif...")
    print("Tekan 'a' untuk simulasi absen.")
    print("Tekan ESC untuk keluar.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Gagal membaca frame kamera.")
                break

            cv2.imshow("IOT-ABSENSI-KEL6 - Face Detection", frame)

            key = cv2.waitKey(1) & 0xFF

            # Ganti kondisi ini dengan hasil face recognition saat sudah tersedia.
            if key == ord("a"):
                result = client.publish(TOPIC, PAYLOAD_ABSEN)
                result.wait_for_publish()
                print("Sinyal Absen Berhasil Dikirim ke ESP32!")

            if key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
