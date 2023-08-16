def pfv():
    import cv2
    import os
    progpath3 = os.path.dirname(os.path.realpath(__file__))
    vidpath3 = os.path.join(progpath3, 'video', 'recordedvid.avi')
    picpath3 = os.path.join(progpath3, 'data', 'current_faces', 'captured')
    vid = cv2.VideoCapture(vidpath3)
    count=0
    success=1
    os.chdir(picpath3)
    i=0
    while success:
        success, img = vid.read()
        cv2.imwrite("frame%d.jpg" %count, img)
        count += 1
        i+=1
        if(i>30):
            break
#pfv()