import cv2 as cv
import mediapipe as mp
import numpy as np
import time
from django.http import StreamingHttpResponse
from django.shortcuts import render
from . import utils  

# Constants for eye indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

CLOSED_EYES_FRAME = 3
FONTS = cv.FONT_HERSHEY_COMPLEX

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh

def landmarksDetection(img, results):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    return mesh_coord

def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = np.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance

def blinkRatio(img, landmarks, right_indices, left_indices):
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance = euclideanDistance(rv_top, rv_bottom)
    lvDistance = euclideanDistance(lv_top, lv_bottom)
    lhDistance = euclideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio

def gen():
    frame_counter = 0
    CEF_COUNTER = 0
    TOTAL_BLINKS = 0

    camera = cv.VideoCapture(0)
    # camera = cv.VideoCapture("http://192.168.5.128:8080/video")

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        start_time = time.time()

        while True:
            frame_counter += 1
            ret, frame = camera.read()
            if not ret:
                break

            frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                mesh_coords = landmarksDetection(frame, results)
                ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

                if ratio > 5.5:
                    CEF_COUNTER += 1
                else:
                    if CEF_COUNTER > CLOSED_EYES_FRAME:
                        TOTAL_BLINKS += 1
                        CEF_COUNTER = 0

                cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, (0, 255, 0), 1, cv.LINE_AA)
                cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, (0, 255, 0), 1, cv.LINE_AA)

                utils.colorBackgroundText(frame, f'Ratio: {round(ratio, 2)}', FONTS, 0.7, (30, 100), 2, (255, 255, 255), (0, 0, 255))
                utils.colorBackgroundText(frame, f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30, 150), 2, (255, 255, 255), (0, 0, 255))

            end_time = time.time() - start_time
            fps = frame_counter / end_time
            frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)

            ret, jpeg = cv.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    camera.release()

def video_feed(request):
    return StreamingHttpResponse(gen(), content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'index.html')
