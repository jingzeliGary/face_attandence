"""
 CNN（resnet)
人脸注册：人脸特征和对应的人脸存入 feature.csv文件
人脸识别：将检测的人脸与 feature.csv文件中的人脸特征 对比，将特征距离相近的人脸名称计入 attendance.csv文件

"""
import csv
import time
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt


def faceRegister(ID, name, count=3, interval=3):
    # 人脸检测 hog
    face_detector = dlib.get_frontal_face_detector()
    # 人脸关键点检测
    shape_detector = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')
    # 人脸特征提取
    feature_descriptor = dlib.face_recognition_model_v1('./weights/dlib_face_recognition_resnet_model_v1.dat')
    # 写入 csv
    csv_file = open('./data/feature.csv', 'a', newline='')
    csv_writer = csv.writer(csv_file)

    collect_count = 0
    start_time = time.time()

    # 读取摄像头
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 检测人脸
        detections = face_detector(img, 1)

        # 解析坐标
        for face in detections:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            # 检测人脸关键点
            points = shape_detector(img, face)
            # 解析关键点坐标
            for point in points.parts():
                cv2.circle(img, (point.x, point.y), 1, (0, 255, 0), -1)

            if collect_count < count:
                cur_time = time.time()
                if cur_time - start_time > interval:
                    # 提取人脸特征
                    feature = feature_descriptor.compute_face_descriptor(img, points)  # 返回特征向量
                    feature = [f for f in feature]  # 转为list

                    # 写入csv
                    line = [ID, name, feature]
                    csv_writer.writerow(line)

                    print(f'采集:{collect_count}'.format(collect_count=collect_count))
                    collect_count += 1
                    start_time = cur_time

                else:
                    pass

            else:
                print('采集结束')
                return

        # 显示图片
        cv2.imshow('Face Recognition', img)

        # 关闭条件
        if cv2.waitKey(10) & 0xFF == 27:
            break

    # 释放
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()


def getFeatureList():
    # 构造列表
    ID_list = []
    name_list = []
    feature_list = []

    with open('./data/feature.csv', 'r') as f:
        csv_reader = csv.reader(f)

        for line in csv_reader:
            ID = line[0]
            name = line[1]
            feature = eval(line[2])  # str-->list
            # print(len(feature))
            feature = np.asarray(feature, dtype=np.float64)
            feature = np.reshape(feature, (1, -1))

            ID_list.append(ID)
            name_list.append(name)
            feature_list.append(feature)
    feature_list = np.concatenate(feature_list, axis=0)
    # print(feature_list.shape)

    return ID_list, name_list, feature_list


def faceRecognizer(threshold=0.5):
    # 人脸检测器
    hog_face_detector = dlib.get_frontal_face_detector()
    # 关键点检测器
    shape_detector = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')
    # 特征提取
    face_descriptor_extractor = dlib.face_recognition_model_v1('./weights/dlib_face_recognition_resnet_model_v1.dat')

    # 读取特征
    ID_list, name_list, feature_list = getFeatureList()
    # 字典记录人脸识别记录
    recog_record = {}
    # CSV写入
    f = open('./data/attendance.csv', 'a', newline="")
    csv_writer = csv.writer(f)
    # 帧率信息
    fps_time = time.time()

    cap = cv2.VideoCapture(0)

    # 获取长宽
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, img = cap.read()

        # 缩放
        img = cv2.resize(img, (width // 2, height // 2))

        # 镜像
        img = cv2.flip(img, 1)

        # 检测人脸
        detections = hog_face_detector(img, 1)
        # 解析坐标
        for face in detections:
            # 人脸框坐标
            l, t, r, b = face.left(), face.top(), face.right(), face.bottom()

            # 检测人脸关键点
            points = shape_detector(img, face)

            # 矩形人脸框
            cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)

            # 提取特征
            feature = face_descriptor_extractor.compute_face_descriptor(img, points)  # 特征向量
            feature = [f for f in feature]  # list
            feature = np.asarray(feature, dtype=np.float64)  # numpy
            feature = feature.reshape((1, -1))

            # 计算与库的距离
            distances = np.linalg.norm((feature - feature_list), axis=1)
            # 最短距离索引
            min_index = np.argmin(distances)
            # 最短距离
            min_distance = distances[min_index]

            if min_distance < threshold:

                predict_id = ID_list[min_index]
                predict_name = name_list[min_index]

                cv2.putText(img, predict_name, (l, b + 40),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

                now = time.time()
                need_insert = False
                # 判断是否识别过
                if predict_name in recog_record:
                    # 存过
                    # 隔一段时间再存
                    if now - recog_record[predict_name] > 3:
                        # 超过阈值时间，再存一次
                        need_insert = True
                        recog_record[predict_name] = now
                    else:
                        # 还没到时间
                        pass
                        need_insert = False
                else:
                    # 没有存过
                    recog_record[predict_name] = now
                    # 存入CSV文件
                    need_insert = True

                if need_insert:
                    time_local = time.localtime(recog_record[predict_name])
                    # 转换格式
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time_local)

                    line = [predict_id, predict_name, min_distance, time_str]
                    csv_writer.writerow(line)
                    print('{time}: 写入成功:{name}'.format(name=predict_name, time=time_str))

            else:
                # print('未识别')
                pass

        # 计算帧率
        now = time.time()
        fps = 1 / (now - fps_time)
        fps_time = now

        cv2.putText(img, "FPS: " + str(round(fps, 2)), (20, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 1)

        # 显示画面
        cv2.imshow('Face attendance', img)

        # 退出条件
        if cv2.waitKey(10) & 0xFF == 27:
            break

    # 释放
    f.close()
    cap.release()
    cv2.destroyAllWindows()


# faceRegister(ID=1, name='Lee')
# getFeatureList()
# faceRecognizer()
