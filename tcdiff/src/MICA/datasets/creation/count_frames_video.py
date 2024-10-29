import sys
import os
import cv2


def count_total_frames_manual(path_video=''):
    video = cv2.VideoCapture(path_video)
    count = 0
    while True:
        (hasNext, frame) = video.read()
        if not hasNext:
            break

        count += 1

    return count

def count_total_frames_auto(path_video=''):
    cap = cv2.VideoCapture(path_video)
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return totalFrames



if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        path_video = '/datasets1/bjgbiesseck/FlorenceFace/Original/subject_01/Video/Indoor-Cooperative.mjpg'


    else:
        path_video = sys.argv[1]
    
    if os.path.isfile(path_video):
        count = count_total_frames_manual(path_video)

        
        print('count:', count, 'frames')
    else:
        print('Error, file not found:', path_video)