import cv2
import mediapipe as mp
import pyttsx3

# === Inisialisasi Mediapipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# === Inisialisasi pyttsx3 ===
engine = pyttsx3.init('sapi5')  # pakai driver Windows
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # 0 = suara cowok, 1 = suara cewek (coba ganti kalau mau)
engine.setProperty('rate', 150)  # kecepatan bicara

last_gesture = None  # supaya tidak spam suara

# === Fungsi untuk mengenali gesture ===
def recognize_gesture(hand_landmarks):
    ujung_jempol = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    ujung_telunjuk = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    ujung_jariTengah = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ujung_jarimanis = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ujung_kelingking = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # HALO (jempol terangkat)
    if (ujung_jempol.y < ujung_telunjuk.y and
        ujung_jempol.y < ujung_jariTengah.y and
        ujung_jempol.y < ujung_jarimanis.y and
        ujung_jempol.y < ujung_kelingking.y):
        return "Halo"
    
    # PERKENALKAN (telunjuk + tengah terangkat)
    if (ujung_telunjuk.y < ujung_jempol.y and
        ujung_jariTengah.y < ujung_jempol.y and
        ujung_jarimanis.y > ujung_jempol.y and
        ujung_kelingking.y > ujung_jempol.y):
        return "Perkenalkan"

    # NAMA SAYA (semua jari lebih tinggi dari jempol)
    if (ujung_jempol.y > ujung_telunjuk.y and
        ujung_jempol.y > ujung_jariTengah.y and
        ujung_jempol.y > ujung_jarimanis.y and
        ujung_jempol.y > ujung_kelingking.y):
        return "Nama saya"

    # SHANDY (telunjuk + kelingking terangkat)
    if (ujung_telunjuk.y < ujung_jempol.y and
        ujung_kelingking.y < ujung_jempol.y and
        ujung_jariTengah.y > ujung_jempol.y and
        ujung_jarimanis.y > ujung_jempol.y):
        return "Shandy"
    
    return None

# === Fungsi deteksi tangan ===
def detect_hand_gesture(image, hand):
    global last_gesture

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hand.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            gesture = recognize_gesture(hand_landmarks)

            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if gesture:
                cv2.putText(image, gesture, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # === Supaya suara hanya keluar kalau gesture berubah ===
                if gesture != last_gesture:
                    engine.say(gesture)
                    engine.runAndWait()
                    last_gesture = gesture

    return image

# === Buka kamera ===
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("tidak dapat membuka kamera")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal menangkap frame")
        break

    frame = detect_hand_gesture(frame, hands)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('w'):
        break

cap.release()
cv2.destroyAllWindows()
