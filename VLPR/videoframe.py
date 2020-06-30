import cv2
import plateDetect as ya

# Take path input
videoName = input('Enter the name of the video with extension: ')

# Opens the Video file
# cap = cv2.VideoCapture(r'C:\Users\ASMA\Desktop\sih2020entire\externalsih2020\video1.mp4')
cap = cv2.VideoCapture(f'input\\{videoName}')

frame_count = 1
i = 0
dictionary = {}
diction_list = []
fps = cap.get(cv2.CAP_PROP_FPS) // 3
try:
    # Passing the images as frames
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if(i % fps == 0):
            print('i-->{}, ret-->{}'.format(i, ret))
            diction, frame_count = ya.yolo(frame, frame_count)
            if bool(diction):
                diction_list.append(diction)
        i+=1
except:
    pass
finally:
    cap.release()
    print('\n')
    j = 1
    for i in diction_list:
        if i[0]['license_plate'] != '' and len(i[0]['license_plate']) >= 6:
            print(f'License Plate No {j}--->> {i[0]["license_plate"]}')
            j += 1
    print('\n')
