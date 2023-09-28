import dense_optical_flow
import cv2

# read test_0.mp4 with opencv
cap = cv2.VideoCapture('test_0.mp4')
ret, old_frame = cap.read()
altitude = 40/3.281  # 40 ft to meters
fov = 120  # fov of camera, in degrees
fps = cap.get(cv2.CAP_PROP_FPS)
pixels_on_axis = 1080  # pixels on y axis, because that's the direction we're moving for test_0.mp4
effective_pixel_cover = 1  # no fisheye/distortion on test_0.mp4

while cap.isOpened():
    ret, new_frame = cap.read()
    if not ret:
        break

    ground_speed = dense_optical_flow.dense_optical_flow(old_frame, new_frame, fov, fps, altitude, pixels_on_axis,
                                                         effective_pixel_cover)
    old_frame = new_frame
    print(f"Ground speed: {ground_speed} m/s")
