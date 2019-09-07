from imageai.Detection import VideoObjectDetection
import os
from matplotlib import pyplot as plt
import cv2

execution_path = os.getcwd()
camera = cv2.VideoCapture(0)

color_index = {
        'person': 'honeydew',
    }


resized = False

def forFrame(frame_number, output_array, output_count, returned_frame):

    plt.clf()

    this_colors = []
    labels = []
    sizes = []

    for eachItem in output_count:
        labels.append(eachItem + " = " + str(output_count[eachItem]))
        sizes.append(output_count[eachItem])
        this_colors.append(color_index['person'])

    global resized

    if (resized == False):
        manager = plt.get_current_fig_manager()
        manager.resize(500, 500)
        resized = True

    plt.subplot(1, 2, 1)
    plt.title("Frame : " + str(frame_number))
    plt.axis("off")
    plt.imshow(returned_frame, interpolation="none")

    plt.subplot(1, 2, 2)
    plt.title("Analysis: " + str(frame_number))
    plt.pie(sizes, labels=labels, colors=this_colors, shadow=True, startangle=140, autopct="%1.1f%%")

    plt.pause(0.01)



video_detector = VideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
video_detector.loadModel()

plt.show()

video_detector.detectObjectsFromVideo(
    camera_input=camera,
    save_detected_video = False,
    frames_per_second=10, per_frame_function=forFrame,
    minimum_percentage_probability=30,
    return_detected_frame=True
    )
