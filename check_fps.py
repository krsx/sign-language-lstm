import cv2
import time

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    new_frame_time = time.time()

    if (new_frame_time - prev_frame_time) != 0:
        fps = 1/(new_frame_time - prev_frame_time)
    else:
        fps = 0

    prev_frame_time = new_frame_time

    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (100, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Webcam FPS Test', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
