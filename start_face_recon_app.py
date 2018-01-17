import sys
import os
import numpy as np
from face_recognition_system.videocamera import VideoCamera
from face_recognition_system.detectors import FaceDetector
import cv2
import shutil
from cv2 import __version__

def resize(images, size=(100, 100)):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm
def cut_face_rectangle(image, face_coord):
    images_rectangle = []
    for (x, y, w, h) in face_coord:
        images_rectangle.append(image[y: y + h, x: x + w])
    return images_rectangle

def draw_face_rectangle(image, faces_coord):
    for (x, y, w, h) in faces_coord:
        cv2.rectangle(image, (x, y), (x + w, y + h), (206, 0, 209), 2)
    return image


def get_images(frame, faces_coord, shape):
    faces_img = cut_face_rectangle(frame, faces_coord)
    frame = draw_face_rectangle(frame, faces_coord)
    #faces_img = normalize_intensity(faces_img)
    faces_img = resize(faces_img)
    return (frame, faces_img)

def add_person(people_folder, shape):
    person_name = input('Name of the new person: ').lower()
    folder = people_folder + person_name
    if os.path.exists(folder):
        shutil.rmtree(folder)
    input("Press ENTER to start taking pictures")
    os.mkdir(folder)
    video = VideoCamera()
    detector = FaceDetector('face_recognition_system/haarcascade_frontalface_alt2.xml')
    counter = 1
    timer = 0
    while counter < 51:
        frame = video.get_frame()
        face_coord = detector.detect(frame)
        if len(face_coord):
            frame, face_img = get_images(frame, face_coord, shape)
            if timer % 10 == 5:
                cv2.imwrite(folder + '/' + str(counter) + '.jpg',
                            face_img[0])
                counter += 1

        cv2.imshow('Video Feed', frame)
        cv2.waitKey(50)
        timer += 5
    cv2.destroyAllWindows()
    
def recognize_people(people_folder, shape):
    people = [person for person in os.listdir(people_folder)]

    print (30 * '-')
    detector = FaceDetector('face_recognition_system/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    threshold = 105
    images = []
    labels = []
    labels_people = {}
    for i, person in enumerate(people):
        labels_people[i] = person
        for image in os.listdir(people_folder + person):
            images.append(cv2.imread(people_folder + person + '/' + image, 0))
            labels.append(i)
    try:
        recognizer.train(images, np.array(labels))
    except:
        print ("\nOpenCV Error: Do you have at least two people in the database?\n")
        sys.exit()

    video = VideoCamera()
    while True:
        frame = video.get_frame()
        faces_coord = detector.detect(frame, False)
        if len(faces_coord):
            frame, faces_img = get_images(frame, faces_coord, shape)
            for i, face_img in enumerate(faces_img):
                if __version__ == "3.1.0":
                    collector = cv2.face.MinDistancePredictCollector()
                    recognizer.predict(face_img, collector)
                    conf = collector.getDist()
                    pred = collector.getLabel()
                else:
                    pred, conf = recognizer.predict(face_img)
                if conf < threshold:
                    cv2.putText(frame, labels_people[pred].capitalize(),
                                (faces_coord[i][0], faces_coord[i][1] - 2),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (206, 0, 209), 2,
                                cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Unknown",
                                (faces_coord[i][0], faces_coord[i][1]),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (206, 0, 209), 2,
                                cv2.LINE_AA)

        cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (206, 0, 209), 2, cv2.LINE_AA)
        cv2.imshow('Video', frame)
        if cv2.waitKey(100) & 0xFF == 27:
            cv2.destroyAllWindows()
            sys.exit()

def check_choice():
    is_valid = 0
    while not is_valid:
        try:
            choice = int(input('Enter your choice [1-3] : '))
            if choice in [1, 2, 3]:
                is_valid = 1
            else:
                print ("'%d' is not an option.\n" % choice)
        except ValueError as error:
            print ("%s is not an option.\n" % str(error).split(": ")[1])
    return choice

if __name__ == '__main__':
    
    print ("1. Add person to the recognizer system")
    print ("2. Start recognizer")
    print ("3. Exit")
    print (30 * '-')

    CHOICE = check_choice()

    PEOPLE_FOLDER = "face_recognition_system/people/"
    SHAPE = "rectangle"

    if CHOICE == 1:
        if not os.path.exists(PEOPLE_FOLDER):
            os.makedirs(PEOPLE_FOLDER)
        add_person(PEOPLE_FOLDER, SHAPE)
    elif CHOICE == 2:
        recognize_people(PEOPLE_FOLDER, SHAPE)
    elif CHOICE == 3:
        sys.exit()
