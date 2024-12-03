import cv2
import mediapipe as mp
import numpy as np
record_file = "results.txt"
try:
    with open(record_file, "r") as f:
        record = float(f.read().strip())
except (FileNotFoundError, ValueError):
    record = 0.0
# Константы
GRID_COLOR = (0, 255, 0)
LINE_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
PATH_COLOR = (255, 0, 0)
POINT_COLOR = (0, 255, 0)
POINT_RADIUS = 4


# Функция для отображения сообщения
def show_message(message, duration=3):
    message_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(message_frame, message, (50, 360), FONT, 1, TEXT_COLOR, 3, cv2.LINE_AA)
    cv2.imshow("Message", message_frame)
    cv2.waitKey(duration * 1000)
    cv2.destroyWindow("Message")


# Функция для отображения обратного отсчета
def draw_countdown(frame, seconds_left):
    text = str(seconds_left)
    text_size = cv2.getTextSize(text, FONT, 3, 3)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), FONT, 3, (0, 0, 255), 3)


# Функция для вычисления точности
def calculate_accuracy(path_points, center_x, center_y, width, height, a, b):
    errors = []
    for x_tip, y_tip in path_points:
        normalized_x = (x_tip - center_x) / 150
        try:
            ideal_y = center_y - a * math.sin(normalized_x + b * math.pi) * 180
            error = abs(y_tip - ideal_y)
            errors.append(error)
        except:
            continue
    if errors:
        median_error = np.mean(errors)
        accuracy = max(0, 100 - median_error)
    else:
        accuracy = 0
    return accuracy


# Основная функция
def run_sin_program():
    global phase, record, record_file
    hands_detector = mp.solutions.hands.Hands(max_num_hands=1)
    cap = cv2.VideoCapture(0)

    a = random.randint(-2, 2)
    if a == 0: a = 1
    b = random.randint(-2, 2)
    c=random.randint(0,1)
    if c == 0:
        show_message(f"Draw y = {a}*sin(x + {b}*pi) from -pi to pi. Prepare for 15 seconds", duration=7)
    else:
        show_message(f"Draw y = {a}*cos(x + {b}*pi) from -pi to pi. Prepare for 15 seconds", duration=7)
        b=b+0.5

    path_points = []
    countdown_time = 8
    start_recording = False
    countdown_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            break

        flipped = np.fliplr(frame).copy()
        flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

        height, width, _ = flipped.shape
        center_x, center_y = width // 2, height // 2

        cv2.line(flipped, (0, center_y), (width, center_y), GRID_COLOR, LINE_THICKNESS)
        cv2.line(flipped, (center_x, 0), (center_x, height), GRID_COLOR, LINE_THICKNESS)
        points = [0, math.pi / 2, math.pi, -math.pi / 2, -math.pi]
        labels = ['0', 'pi/2', 'pi', '-pi/2', '-pi']
        for i, point in enumerate(points):
            pixel_x = center_x + int(point * width / (2 * 4 / 3 * math.pi)) - 25
            cv2.putText(flipped, labels[i], (pixel_x + 10, center_y + 20), FONT, 0.5, TEXT_COLOR, FONT_THICKNESS)
            cv2.circle(flipped, (pixel_x + 25, center_y), POINT_RADIUS, POINT_COLOR, -1)
        cv2.putText(flipped, "1", (center_x, center_y - 180), FONT, 0.5, TEXT_COLOR, FONT_THICKNESS)
        cv2.putText(flipped, "-1", (center_x, center_y + 180), FONT, 0.5, TEXT_COLOR, FONT_THICKNESS)
        cv2.circle(flipped, (center_x, center_y - 170), POINT_RADIUS, POINT_COLOR, -1)
        cv2.circle(flipped, (center_x, center_y + 170), POINT_RADIUS, POINT_COLOR, -1)
        cv2.putText(flipped, "2", (center_x, center_y - 330), FONT, 0.5, TEXT_COLOR, FONT_THICKNESS)
        cv2.putText(flipped, "-2", (center_x, center_y + 330), FONT, 0.5, TEXT_COLOR, FONT_THICKNESS)
        cv2.circle(flipped, (center_x, center_y - 320), POINT_RADIUS, POINT_COLOR, -1)
        cv2.circle(flipped, (center_x, center_y + 320), POINT_RADIUS, POINT_COLOR, -1)

        elapsed_time = time.time() - countdown_start
        if not start_recording and elapsed_time < countdown_time:
            draw_countdown(flipped, countdown_time - int(elapsed_time))
        elif not start_recording:
            start_recording = True
            start_time = time.time()

        results = hands_detector.process(flippedRGB)
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                x_tip = int(hand_landmarks.landmark[8].x * width)
                y_tip = int(hand_landmarks.landmark[8].y * height)
                cv2.circle(flipped, (x_tip, y_tip), 10, PATH_COLOR, -1)

                if start_recording:
                    path_points.append((x_tip, y_tip))
                if start_recording and (x_tip < 0 or x_tip > width / 2 + width / 3 + 50):
                    end_time = time.time()
                    duration = end_time - start_time

                    accuracy = calculate_accuracy(path_points, center_x, center_y, width, height, a, b)

                    # Отображаем результат на экране
                    result_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(result_frame, f"Duration: {duration:.2f} seconds", (50, 300), FONT, 2, (255, 255, 255),
                                5, cv2.LINE_AA)
                    cv2.putText(result_frame, f"Accuracy: {accuracy:.2f}%", (50, 400), FONT, 2, (255, 255, 255), 5,
                                cv2.LINE_AA)


                    if accuracy > record:
                        record = accuracy
                        with open(record_file, "w") as f:
                            f.write(f"{record:.2f}")
                        cv2.putText(result_frame, "New Record!", (50, 500), FONT, 2, (0, 255, 0), 5, cv2.LINE_AA)
                    else:
                        cv2.putText(result_frame, f"Record: {record:.2f}%", (50, 500), FONT, 2, (255, 255, 255), 5,
                                    cv2.LINE_AA)
                    cv2.imshow("Result", result_frame)
                    cv2.waitKey(5000)  # Показ результата в течение 5 секунд
                    cap.release()
                    cv2.destroyAllWindows()

        if len(path_points) > 1:
            for i in range(1, len(path_points)):
                cv2.line(flipped, path_points[i - 1], path_points[i], PATH_COLOR, 2)

        res_image = cv2.cvtColor(flipped, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hands", res_image)

    hands_detector.close()
    cap.release()
    cv2.destroyAllWindows()
    phase = 1






import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random

# Создаем детектор рук
handsDetector = mp.solutions.hands.Hands(max_num_hands=1)

# Константы
GRID_COLOR = (0, 255, 0)
LINE_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)

# Координаты кнопок в меню
buttons_menu = {
    "start": ((400, 300), (600, 400)),
    "exit": ((400, 450), (600, 550)),
}
buttons_restart = {
    "restart": ((400, 300), (600, 400)),
    "menu": ((400, 450), (600, 550)),
}

phase = 0
# Функция для отрисовки меню
def draw_buttons(frame, buttons):
    for name, ((x1, y1), (x2, y2)) in buttons.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, name.upper(), (x1 + 20, y1 + 60), FONT, 1, TEXT_COLOR, 2)


# Заглушка для кнопки cos
def placeholder():
    placeholder_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(placeholder_frame, "Заглушка", (450, 360), FONT, 2, TEXT_COLOR, 3)
    cv2.imshow("Cos Placeholder", placeholder_frame)
    cv2.waitKey(2000)
    cv2.destroyWindow("Cos Placeholder")



def mouse_callback(event, x, y, flags, param):
    global phase
    if event == cv2.EVENT_LBUTTONDOWN:
        if phase == 0:  # Главное меню
            for name, ((x1, y1), (x2, y2)) in buttons_menu.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if name == "start":
                        phase = 2  # Переход к игре
                    elif name == "exit":
                        cv2.destroyAllWindows()
                        exit()
        elif phase == 1:  # Меню перезапуска
            for name, ((x1, y1), (x2, y2)) in buttons_restart.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if name == "restart":
                        phase = 2  # Переход к игре
                    elif name == "menu":
                        phase = 0  # Переход в главное меню
cv2.setMouseCallback("Main Window", mouse_callback)

while True:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    if phase == 0:  # Главное меню
        cv2.putText(frame, f"Record: {record:.2f}%", (450, 150), FONT, 2, TEXT_COLOR, 3)

        draw_buttons(frame, buttons_menu)
    elif phase == 1:  # Меню перезапуска
        cv2.putText(frame, "Restart Menu", (450, 150), FONT, 2, TEXT_COLOR, 3)
        draw_buttons(frame, buttons_restart)
    elif phase == 2:  # Игра
        run_sin_program()
        continue

    # Отображение окна
    cv2.imshow("Main Window", frame)
    cv2.setMouseCallback("Main Window", mouse_callback)  # Регистрация обработчика событий

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

