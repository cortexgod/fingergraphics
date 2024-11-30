import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random
# Создаем детектор рук
handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)

# Константы
grid_color = (0, 255, 0)
line_thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX
font_thickness = 2
text_color = (255, 255, 255)
path_color = (255, 0, 0)
point_color = (0, 255, 0)
point_radius = 4

# Функция для отображения обратного отсчета
def draw_countdown(frame, seconds_left):
    text = str(seconds_left)
    text_size = cv2.getTextSize(text, font, 3, 3)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, 3, (0, 0, 255), 3)

# Функция для вычисления точности
def calculate_accuracy(path_points, center_x, center_y, width, height, a, b):
    errors = []
    for x_tip, y_tip in path_points:
        # Нормализация координаты X
        normalized_x = (x_tip - center_x) / 150  
        try:
            # Вычисление идеальной Y-координаты с учетом новой формулы
            ideal_y = center_y - a * math.sin(normalized_x + b * math.pi) * 180  
            error = abs(y_tip - ideal_y)
            errors.append(error)
        except:
            continue  # Игнорируем разрывы функции

    if errors:
        median_error = np.mean(errors)
        accuracy = max(0, 100 - median_error)  # Адаптация метрики
    else:
        accuracy = 0

    return accuracy

# В начале программы выводим сообщение
message_frame = np.zeros((720, 1280, 3), dtype=np.uint8)



a = random.randint(-2, 2)
if a==0: a=1
b = random.randint(-2, 2)
cv2.putText(message_frame, f"Draw a y = {a}*sin(x+{b}*pi) graph from -pi to pi, you will have 15 second to prepare", (50, 360), 
            font, 1, (255, 255, 255),3, cv2.LINE_AA)
cv2.imshow("Message", message_frame)
cv2.waitKey(7000)  # Показываем сообщение в течение 3 секунд
cv2.destroyWindow("Message")

# Переменные для записи пути пальца
path_points = []
countdown_time = 8  # Время для обратного отсчета
start_recording = False
countdown_start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
        break
    flipped = np.fliplr(frame).copy()
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

    # Рисуем оси
    height, width, _ = flipped.shape
    center_x, center_y = width // 2, height // 2
    cv2.line(flipped, (0, center_y), (width, center_y), grid_color, line_thickness)
    cv2.line(flipped, (center_x, 0), (center_x, height), grid_color, line_thickness)

    # Добавляем метки осей
    points = [0, math.pi/2, math.pi, -math.pi/2, -math.pi]
    labels = ['0', 'pi/2', 'pi', '-pi/2', '-pi']
    for i, point in enumerate(points):
        pixel_x = center_x + int(point * width / (2 * 4 / 3 * math.pi)) - 25
        cv2.putText(flipped, labels[i], (pixel_x + 10, center_y + 20), font, 0.5, text_color, font_thickness)
        cv2.circle(flipped, (pixel_x + 25, center_y), point_radius, point_color, -1)
    cv2.putText(flipped, "1", (center_x, center_y - 180), font, 0.5, text_color, font_thickness)
    cv2.putText(flipped, "-1", (center_x, center_y + 180), font, 0.5, text_color, font_thickness)
    cv2.circle(flipped, (center_x, center_y - 170), point_radius, point_color, -1)
    cv2.circle(flipped, (center_x, center_y + 170), point_radius, point_color, -1)
    cv2.putText(flipped, "2", (center_x, center_y - 330), font, 0.5, text_color, font_thickness)
    cv2.putText(flipped, "-2", (center_x, center_y + 330), font, 0.5, text_color, font_thickness)
    cv2.circle(flipped, (center_x, center_y - 320), point_radius, point_color, -1)
    cv2.circle(flipped, (center_x, center_y + 320), point_radius, point_color, -1)
    

    # Отображаем обратный отсчёт
    elapsed_time = time.time() - countdown_start
    if not start_recording and elapsed_time < countdown_time:
        draw_countdown(flipped, countdown_time - int(elapsed_time))
    elif not start_recording:
        start_recording = True
        start_time = time.time()  # Начало записи пути

    # Обработка рук
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        for hand_landmarks in results.multi_hand_landmarks:
            x_tip = int(hand_landmarks.landmark[8].x * width)
            y_tip = int(hand_landmarks.landmark[8].y * height)
            cv2.circle(flipped, (x_tip, y_tip), 10, (255, 0, 0), -1)

            if start_recording:
                path_points.append((x_tip, y_tip))

            if start_recording and (x_tip < 0 or x_tip > width/2+width/3+50):
                end_time = time.time()
                duration = end_time - start_time

                accuracy = calculate_accuracy(path_points, center_x, center_y, width, height, a, b)

                # Отображаем результат на экране
                result_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(result_frame, f"Duration: {duration:.2f} seconds", (50, 300), font, 2, (255, 255, 255), 5, cv2.LINE_AA)
                cv2.putText(result_frame, f"Accuracy: {accuracy:.2f}%", (50, 400), font, 2, (255, 255, 255), 5, cv2.LINE_AA)
                cv2.imshow("Result", result_frame)
                cv2.waitKey(5000)  # Показ результата в течение 5 секунд
                cap.release()
                cv2.destroyAllWindows()
                exit()

    # Рисуем путь
    if len(path_points) > 1:
        for i in range(1, len(path_points)):
            cv2.line(flipped, path_points[i - 1], path_points[i], path_color, 2)

    res_image = cv2.cvtColor(flipped, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

# Освобождаем ресурсы
handsDetector.close()
cap.release()
cv2.destroyAllWindows()
