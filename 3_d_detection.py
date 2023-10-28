import cv2
import mediapipe as mp
import time

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

objectron = mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.8,
    model_name="Cup",
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    start = time.time()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = objectron.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.detected_objects:  
        print(results.detected_objects)
        for objects in results.detected_objects:
            print(objects)
            mp_drawing.draw_landmarks(
                frame, objects.landmarks_2d, mp_objectron.BOX_CONNECTIONS
            )
            mp_drawing.draw_axis(
                frame, objects.rotation, objects.translation
            )

    end = time.time()
    t_time = end - start
    fps = 1 / t_time
    cv2.putText(
        frame,
        f"FPS:{int(fps)}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        2,
    )
    cv2.imshow("detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
