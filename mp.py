# IMPORTING LIBRARIES
from cv2 import cv2
import mediapipe as mp

# INITIALIZING OBJECTS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

# DETECT THE FACE LANDMARKS
with mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.8, min_tracking_confidence=0.8) as face_mesh, \
        mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while True:
        success, image = cap.read()

        # Flip the image horizontally and convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Detect the face landmarks
        results_face = face_mesh.process(image)
        results_hands = hands.process(image)

        # To improve performance
        image.flags.writeable = True

        # Convert back to the BGR color space
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the face mesh annotations on the image.
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Display the image
        cv2.imshow('MediaPipe FaceMesh and Hands', image)

        # Terminate the process
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()