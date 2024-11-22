import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Создаем детектор рук
handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)

# Константы
grid_color = (0, 255, 0)
line_thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1
text_color = (255, 255, 255)
path_color = (255, 0, 0)  # Цвет для следа (синий)
ideal_line_color = (0, 255, 255)  # Цвет идеальной линии (желтый)

# Функция для отображения обратного отсчета
def draw_countdown(frame, seconds_left):
    text = str(seconds_left)
    text_size = cv2.getTextSize(text, font, 3, 3)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, 3, (0, 0, 255), 3)

# Переменные для записи пути пальца
path_points = []
countdown_time = 7  # Время для обратного отсчета
start_recording = False
countdown_start = time.time()

# Функция для рисования идеальной функции тангенса
def draw_ideal_tangent(frame, center_x, width, height):
    # Рисуем идеальную функцию тангенса (y = tan(x))
    for i in range(-8, 8):
        x = i * width // 16  # Расчет X
        normalized_x = (x - center_x) / (width // 16)  # Нормализуем X для функции тангенса
        try:
            y = math.tan(normalized_x) * height // 4  # Расчет Y для функции тангенса
            pixel_y = int(center_y - y)  # Инвертируем координату Y
            cv2.circle(frame, (x + center_x, pixel_y), 1, ideal_line_color, -1)
        except:
            continue  # Игнорируем значения за пределами области (например, в точках разрыва)

# Функция для вычисления точности
import math

def calculate_accuracy(path_points, center_x, center_y, width, height):
    errors = []
    for x_tip, y_tip in path_points:
        # Нормализуем x координату
        normalized_x = (x_tip - center_x) / 150
        try:
            # Вычисляем идеальный y
            ideal_y = 360 - math.sin(normalized_x)*180
            (x_tip, normalized_x, y_tip, ideal_y)
            error = abs(y_tip - ideal_y)
            errors.append(error)
        except:
            continue  # Игнорируем разрывы функции (например, для тангенса)

    # Вычисляем медиану ошибок
    
    if errors:
        median_error = np.median(errors)
        accuracy = max(0, 100 - median_error)  # Пример адаптации метрики
    else:
        accuracy = 0

    return accuracy


while cap.isOpened():
    ret, frame = cap.read()
    if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Отражаем изображение горизонтально
    flipped = np.fliplr(frame).copy()
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

    # Рисуем оси
    height, width, _ = flipped.shape
    center_x, center_y = width // 2, height // 2

    # Рисуем горизонтальную ось
    cv2.line(flipped, (0, center_y), (width, center_y), grid_color, line_thickness)
    # Рисуем вертикальную ось
    cv2.line(flipped, (center_x, 0), (center_x, height), grid_color, line_thickness)

    # Отображаем надписи над нужными точками
    points = [0, math.pi/2, math.pi, -math.pi/2, -math.pi]  # Позиции для надписей
    labels = ['0', 'pi/2', 'pi', '-pi/2', '-pi']  # Текст для каждой точки
    point_color = (0, 255, 0)  # Цвет точки (зелёный)
    point_radius = 4 
    for i, point in enumerate(points):
        pixel_x = center_x + int(point * width / (2*4/3 * math.pi))-25
        cv2.putText(flipped, labels[i], (pixel_x + 10, center_y + 20), font, 0.5, text_color, font_thickness)
        cv2.circle(flipped, (pixel_x+25, center_y ), point_radius, point_color, -1)
    cv2.putText(flipped, "1", (center_x, center_y - 180), font, 0.5, text_color, font_thickness)
    cv2.putText(flipped, "-1", (center_x, center_y + 180), font, 0.5, text_color, font_thickness)
    # Нарисовать точк
    # Радиус точки
    cv2.circle(flipped, (center_x, center_y - 160), point_radius, point_color, -1)
    cv2.circle(flipped, (center_x, center_y + 160), point_radius, point_color, -1)

    
    # Рисуем идеальную функцию тангенса

    # Определяем оставшееся время для обратного отсчета
    elapsed_time = time.time() - countdown_start
    if not start_recording and elapsed_time < countdown_time:
        draw_countdown(flipped, countdown_time - int(elapsed_time))
    elif not start_recording:
        start_recording = True
        start_time = time.time()  # Начало записи пути

    # Распознаем руки
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        for hand_landmarks in results.multi_hand_landmarks:
            x_tip = int(hand_landmarks.landmark[8].x * width)
            y_tip = int(hand_landmarks.landmark[8].y * height)
            cv2.circle(flipped, (x_tip, y_tip), 10, (255, 0, 0), -1)

            # Если запись началась, сохраняем координаты пальца
            if start_recording:
                path_points.append((x_tip, y_tip))

            # Прерываем запись, если палец выходит за границы
            if start_recording and (x_tip < 0 or x_tip > width or y_tip < 0 or y_tip > height):
                end_time = time.time()
                duration = end_time - start_time

                # Рассчитываем точность
                accuracy = calculate_accuracy(path_points, center_x, center_y, width, height)

                print(f"Path recorded for {duration:.2f} seconds")
                print(f"Accuracy: {accuracy:.2f}%")
                cap.release()
                cv2.destroyAllWindows()
                exit()

    # Рисуем след за пальцем
    if len(path_points) > 1:
        for i in range(1, len(path_points)):
            cv2.line(flipped, path_points[i-1], path_points[i], path_color, 2)

    # Преобразуем изображение обратно в BGR для отображения
    res_image = cv2.cvtColor(flipped, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

# Освобождаем ресурсы
handsDetector.close()
cap.release()
cv2.destroyAllWindows()
