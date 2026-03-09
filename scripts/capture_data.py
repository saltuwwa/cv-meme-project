import cv2
import os

# Введи название класса, который собираешь в этот момент
label = input("Enter class name (sleep/salam/hmm/cat-tongue): ")
save_path = f"../dataset/{label}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = len(os.listdir(save_path))
frame_count = 0

print("Press 'S' to save a photo, 'ESC' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Camera", frame)
    frame_count += 1

    # Автосохранение каждые 10 кадров
    if frame_count % 10 == 0:
        cv2.imwrite(f"{save_path}/{count}.jpg", frame)
        count += 1

    key = cv2.waitKey(1)
    if key == ord('s'):  # ручное сохранение
        cv2.imwrite(f"{save_path}/{count}.jpg", frame)
        print(f"Saved: {save_path}/{count}.jpg")
        count += 1
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()