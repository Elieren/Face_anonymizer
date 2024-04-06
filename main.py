from multiprocessing import Process

import pyaudio
import numpy as np
import random

from ultralytics import YOLO
import cv2
import pyvirtualcam

import sys  # sys нужен для передачи argv в QApplication
from PyQt5 import QtWidgets

import design

import sounddevice as sd

import py3nvml.py3nvml as nvidia

import qdarktheme


def audio(hz, hz_min, hz_max, input_dev, output_dev, audio_on):
    # Создаем объект PyAudio
    p = pyaudio.PyAudio()

    # Открываем поток для чтения данных с микрофона
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=44100,
                    input=True,
                    output=True,
                    input_device_index=input_dev,
                    output_device_index=output_dev,
                    frames_per_buffer=1024)

    x = 0
    gain = 3.0

    while True:
        if x == 10:
            hz = random.randint(hz_min, hz_max)
            x = 0
            # pass

        # Читаем данные из потока
        data = np.fromstring(stream.read(1024), dtype=np.float32)

        if audio_on:
            # Применяем преобразование Фурье, чтобы получить частоты
            fft = np.fft.rfft(data)

            # Сдвигаем частоты на 1000 Гц
            fft = np.roll(fft, hz)

            # Применяем обратное преобразование Фурье
            data = np.fft.irfft(fft)

            data = data * gain

        # window = np.hanning(len(data))
        # data = data * window

        x += 1

        # Воспроизводим обработанные данные
        stream.write(data.astype(np.float32).tostring())


def video(model, dev, size, video_on, video_status, device):

    imgsz = 640
    conf = 0.1
    iou = 0.5

    cap = cv2.VideoCapture(dev)

    if video_status:
        cam = pyvirtualcam.Camera(width=int(size[0]), height=int(size[1]),
                                  fps=20)
    while True:
        success, img = cap.read()
        if not success:
            break

        if video_on:

            results = model.predict(
                img,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False
            )

            # Пройдитесь по всем боксам
            for box in results[0].boxes:
                xyxy = box.xyxy[0]
                x1 = int(xyxy[0] + 0.5)
                y1 = int(xyxy[1] + 0.5)
                x2 = int(xyxy[2] + 0.5)
                y2 = int(xyxy[3] + 0.5)

                img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2],
                                                     (45, 45), 0)

        # Отобразите изображение
        if video_status:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.flip(img, 1)
            cam.send(img)
            cam.sleep_until_next_frame()
        else:
            cv2.imshow('Image', img)
            cv2.waitKey(1)

    if video_status:
        cam.close()
    cv2.destroyAllWindows()


class MainApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        self.start_status = True
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна

        self.model = False

        self.mic_list()
        self.speakers_list()
        self.get_camera_info()
        self.get_gpu_info()

        self.comboBox.addItems([item[1] for item in self.input_devices])
        self.comboBox_2.addItems([item[1] for item in self.output_devices])
        self.comboBox_3.addItems(
            [(str(item[0]) + ' ' + str(item[1])) for item in self.camera_info])
        self.comboBox_4.addItems([item[1] for item in self.videocard_list])

        self.pushButton_3.clicked.connect(self.browse_file_model)

        self.pushButton.clicked.connect(self.start)
        self.pushButton_2.clicked.connect(self.stop)

        self.pushButton_2.setEnabled(False)

    def mic_list(self):
        devices = sd.query_devices()

        self.input_devices = [
            [x, device['name']] for x, device in enumerate(devices) if (
                device['max_input_channels'] > 0) and (device['hostapi'] == 0)]

    def speakers_list(self):
        devices = sd.query_devices()

        self.output_devices = [
            [x, device['name']] for x, device in enumerate(devices) if (
                device['max_output_channels'] > 0
                ) and (device['hostapi'] == 0)]

    def get_camera_info(self):
        self.camera_info = []
        for i in range(10):  # Проверяем первые 10 индексов
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                self.camera_info.append([i, (width, height)])
                cap.release()

    def get_gpu_info(self):
        self.videocard_list = []
        self.videocard_list.append(['cpu', 'CPU'])
        try:
            nvidia.nvmlInit()
            device_count = nvidia.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = nvidia.nvmlDeviceGetHandleByIndex(i)
                info = nvidia.nvmlDeviceGetMemoryInfo(handle)
                total_memory_in_gb = info.total / (1024 ** 3)
                self.videocard_list.append(
                    [i, f"{nvidia.nvmlDeviceGetName(handle)}, \
Memory: {total_memory_in_gb} GB"])
        except Exception:
            pass

    def browse_file_model(self):
        self.checkBox.setChecked(False)
        self.file_model = QtWidgets.QFileDialog.getOpenFileName(
            self, "Выберите файл", "", ("Model Files (*.pt)"))

        if self.file_model[0] != '':
            self.model = YOLO(self.file_model[0])
            self.pushButton_3.setStyleSheet("background-color: white")
            self.checkBox.setChecked(True)

    def video_status(self, video_status, input_video, size_dev, device):
        if self.radioButton_5.isChecked():
            self.p2 = Process(target=video, args=(self.model, input_video,
                                                  size_dev, video_status,
                                                  False, device))
        elif self.radioButton_6.isChecked():
            self.p2 = Process(target=video, args=(self.model, input_video,
                                                  size_dev, video_status,
                                                  True, device))

    def start(self):
        self.pushButton.setStyleSheet("background-color: green")
        self.pushButton_2.setStyleSheet("background-color: white")
        self.pushButton_2.setEnabled(True)
        self.pushButton.setEnabled(False)

        value = self.comboBox.currentIndex()
        input_dev = self.input_devices[value][0]

        value = self.comboBox_2.currentIndex()
        output_dev = self.output_devices[value][0]

        input_video = self.comboBox_3.currentIndex()
        size_dev = self.camera_info[input_video][1]

        device = self.comboBox_4.currentIndex()
        device = self.videocard_list[device][0]

        self.groupBox.setEnabled(False)
        self.groupBox_2.setEnabled(False)
        self.groupBox_3.setEnabled(False)
        self.groupBox_4.setEnabled(False)
        self.groupBox_5.setEnabled(False)
        self.groupBox_6.setEnabled(False)
        self.groupBox_7.setEnabled(False)
        self.groupBox_8.setEnabled(False)
        self.groupBox_9.setEnabled(False)

        if self.radioButton_2.isChecked():
            if self.checkBox.isChecked() and (self.model is not False):
                self.pushButton_3.setStyleSheet("background-color: white")
                self.video_status(True, input_video, size_dev, device)
            else:
                if self.model is False:
                    self.pushButton_3.setStyleSheet("background-color: red")

                self.pushButton.setStyleSheet("background-color: white")
                self.pushButton.setEnabled(True)
                self.pushButton_2.setEnabled(False)

                self.groupBox.setEnabled(True)
                self.groupBox_2.setEnabled(True)
                self.groupBox_3.setEnabled(True)
                self.groupBox_4.setEnabled(True)
                self.groupBox_5.setEnabled(True)
                self.groupBox_6.setEnabled(True)
                self.groupBox_7.setEnabled(True)
                self.groupBox_8.setEnabled(True)
                self.groupBox_9.setEnabled(True)

        elif self.radioButton.isChecked():
            self.pushButton_3.setStyleSheet("background-color: white")
            self.video_status(False, input_video, size_dev, device)

        if self.radioButton_4.isChecked():
            hz = self.spinBox.value()
            hz_min = self.spinBox_2.value()
            hz_max = self.spinBox_3.value()
            self.p1 = Process(target=audio, args=(hz, hz_min, hz_max,
                                                  input_dev, output_dev, True))

        elif self.radioButton_3.isChecked():
            hz = self.spinBox.value()
            hz_min = self.spinBox_2.value()
            hz_max = self.spinBox_3.value()
            self.p1 = Process(target=audio, args=(hz, hz_min, hz_max,
                                                  input_dev, output_dev,
                                                  False))

        try:
            self.p1.start()
        except Exception:
            pass
        try:
            self.p2.start()
        except Exception:
            pass

    def stop(self):
        self.pushButton.setStyleSheet("background-color: white")
        self.pushButton_2.setStyleSheet("background-color: red")
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(False)

        self.groupBox.setEnabled(True)
        self.groupBox_2.setEnabled(True)
        self.groupBox_3.setEnabled(True)
        self.groupBox_4.setEnabled(True)
        self.groupBox_5.setEnabled(True)
        self.groupBox_6.setEnabled(True)
        self.groupBox_7.setEnabled(True)
        self.groupBox_8.setEnabled(True)
        self.groupBox_9.setEnabled(True)

        try:
            self.p1.terminate()
        except Exception:
            pass
        try:
            self.p2.terminate()
        except Exception:
            pass


if __name__ == '__main__':
    qdarktheme.enable_hi_dpi()
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    qdarktheme.setup_theme(custom_colors={"primary": "#00ff00"})
    window = MainApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение
