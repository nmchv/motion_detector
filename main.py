
import cv2
import datetime
import time
import os

if not os.path.exists('output'):
    os.makedirs('output')

cap = cv2.VideoCapture(1)  # 0 - iphone, 1 - mac
if not cap.isOpened():
    print('Проблемы с камерой')
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video_filename = os.path.join("output", f'video_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4')
out = cv2.VideoWriter(video_filename, fourcc, 20, (frame_width, frame_height))

_, frame1 = cap.read()
_, frame2 = cap.read()

start_time = time.time()

while cap.isOpened():
    frame1 = cv2.flip(frame1, 1)
    frame2 = cv2.flip(frame2, 1)

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(threshold, None, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, None)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    elapsed_time = int(time.time() - start_time)
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    timer_text = f"Camera have been open for {minutes:02}:{seconds:02}"
    now = datetime.datetime.now().strftime('Date: %d-%m-%Y Time: %H:%M:%S')
    (now_width, _), _ = cv2.getTextSize(now, cv2.FONT_ITALIC, 0.7, 2)
    cv2.putText(frame1, timer_text, (10, frame_height - 10), cv2.FONT_ITALIC, 0.7, (0, 0, 0), 2)
    cv2.putText(frame1, now, (frame_width - now_width-10, frame_height - 10), cv2.FONT_ITALIC, 0.7, (0, 0, 0), 2)

    cv2.imshow('Motion Detection', frame1)
    out.write(frame1)

    ret, frame1 = cap.read()
    if not ret:
        break
    ret, frame2 = cap.read()
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
