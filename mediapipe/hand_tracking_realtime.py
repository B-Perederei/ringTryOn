import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v  # Avoid division by zero


cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                mcp = np.array([
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z
                ])
                pip = np.array([
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z
                ])
                dip = np.array([
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z
                ])

                Y_finger = normalize(pip - mcp)
                Z_finger = normalize(np.cross(dip - pip, pip - mcp))
                X_finger = np.cross(Y_finger, Z_finger)

                interp_factor = 0.7
                pos = (1 - interp_factor) * mcp + interp_factor * pip
                x_pixel = int(pos[0] * w)
                y_pixel = int(pos[1] * h)

                cv2.circle(frame, (x_pixel, y_pixel), 15, (255, 0, 0), -1)
                cv2.circle(frame, (x_pixel, y_pixel), 10, (0, 0, 255), -1)

                print(f"Ring Placement (Normalized): ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
                print("Finger Coordinate System:")
                print(f"X_finger: {X_finger}")
                print(f"Y_finger: {Y_finger}")
                print(f"Z_finger: {Z_finger}")

                R = np.column_stack((X_finger, Y_finger, Z_finger))
                print("Rotation Matrix (Finger -> Camera):")
                print(R)

                transform_matrix = np.eye(4)
                transform_matrix[:3, 0] = Y_finger
                transform_matrix[:3, 1] = X_finger
                transform_matrix[:3, 2] = -Z_finger
                transform_matrix[:3, 3] = pos

                print("Transformation Matrix:")
                print(transform_matrix)

                pos_homogeneous = np.append(pos, 1)
                camera_pose = np.matmul(transform_matrix, pos_homogeneous)
                print("Camera Pose:")
                print(camera_pose)

        cv2.imshow('Hand Tracking with Coordinate System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
