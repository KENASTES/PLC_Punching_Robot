import cv2

def test_webcam_rotation(camera_index=0):

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    rotation_mode = 0 

    while True:
        success, frame = cap.read()
        
        if not success:
            print("อ่านภาพจากกล้องไม่ได้")
            break

        if rotation_mode == 1:
            display_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_mode == 2:
            display_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_mode == 3:
            display_frame = cv2.rotate(frame, cv2.ROTATE_180)
        else:
            display_frame = frame.copy() 
        cv2.imshow("Webcam Rotation Test", display_frame)

        # รอรับการกดปุ่มจากคีย์บอร์ด (1 ms)
        key = cv2.waitKey(1) & 0xFF

    cap.release()
    cv2.destroyAllWindows()