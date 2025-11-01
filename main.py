import cv2
import numpy as np
from art import tprint


def apply_yolo_object_detection(image_to_process):
    """
    Распознавание и определение координат объектов на изображении
    :param image_to_process: исходное изображение
    :return: изображение с выделенными объектами и подписями к ним
    """

    height, width, _ = image_to_process.shape 
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608),
                                 (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0

    CONF_THRESH = 0.25
    # Поиск объектов на картинке
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]

            if class_score > CONF_THRESH:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)

                # Фильтр слишком маленьких боксов
                min_area = 20 * 20 
                if obj_width * obj_height < min_area:
                    continue

                box = [center_x - obj_width // 2, center_y - obj_height // 2,
                       obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    idxs = cv2.dnn.NMSBoxes(boxes, class_scores, CONF_THRESH, 0.5)

    if len(idxs) > 0:
        for bi in idxs:
            i = int(bi) if isinstance(bi, (int, np.integer)) else int(bi[0])

            # проверяем интересующий класс
            if classes[class_indexes[i]] in classes_to_look_for:
                objects_count += 1
                image_to_process = draw_object_bounding_box(
                    image_to_process,
                    class_indexes[i],
                    boxes[i],
                    class_scores[i]
                )


    final_image = draw_object_count(image_to_process, objects_count)
    return final_image

def draw_object_bounding_box(image_to_process, index, box, confidence):
    """
    Отрисовка рамки и подписи внутри верхней части рамки
    """
    x, y, w, h = box
    label = f"{classes[index]} {confidence:.2f}"

    # Цвет рамки зелёный
    color = (60, 220, 100)
    thickness = 2
    cv2.rectangle(image_to_process, (x, y), (x + w, y + h), color, thickness)

    # Настройки текста
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    # Размер текста
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Ключ: текст внутри рамки, немного ниже верхней границы
    text_x = x + 3
    text_y = y + text_h + 4 

    overlay = image_to_process.copy()
    cv2.rectangle(
        overlay,
        (text_x - 2, text_y - text_h - 2),
        (text_x + text_w + 2, text_y + 2),
        (40, 40, 40),
        -1
    )
    cv2.addWeighted(overlay, 0.6, image_to_process, 0.4, 0, image_to_process)

    # Белый текст
    cv2.putText(image_to_process, label, (text_x, text_y),
                font, font_scale, (230, 230, 230), font_thickness, cv2.LINE_AA)

    return image_to_process



def draw_object_count(image_to_process, objects_count):
    """
    Подпись количества найденных объектов на изображении
    """

    start = (10, 120)
    font_size = 1.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3
    text = "Objects found: " + str(objects_count)

    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)

    return final_image


def start_video_object_detection(video: str):
    """Захват, анализ и сохранение видео с применением YOLO"""
    try:
        # Открываем источник
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"[Ошибка] Не удалось открыть видео: {video}")
            return

        # Получаем параметры входного видео
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:  
            fps = 25.0

        # Подготовка записи выходного видео
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))
        print("[INFO] Нажмите 'q' для выхода или Ctrl+C для прерывания.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Видео закончилось.")
                break

            # Обработка кадра через YOLO
            processed_frame = apply_yolo_object_detection(frame)

            # Запись обработанного кадра в файл
            out.write(processed_frame)

            # Отображение окна
            display_frame = cv2.resize(processed_frame, (960, 540))
            cv2.imshow("Video Capture", display_frame)

            # Выход по клавише 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Завершено пользователем (клавиша q).")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Остановка по Ctrl+C")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("[INFO] Ресурсы освобождены, видео сохранено в 'output.avi'")


if __name__ == '__main__':

    tprint("Object detection")

    # Загружаем архитектуру (cfg) и веса (weights) YOLOv4-tiny
    net = cv2.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg",
                                     "Resources/yolov4-tiny.weights")
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    with open("Resources/coco.names.txt") as file:
        classes = file.read().split("\n")

    #Путь к видео и список интересующих классов
    video = input("Path to video (or URL): ")
    look_for = input("What we are looking for: ").split(',')
    
    # Очищаем пробелы, получаем список искомых строк-классов
    list_look_for = []
    for look in look_for:
        list_look_for.append(look.strip())

    classes_to_look_for = list_look_for

    start_video_object_detection(video)