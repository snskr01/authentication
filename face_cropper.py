def fc():    
    import cv2
    import sys
    import os
    progpath4 = os.path.dirname(os.path.realpath(__file__))
    #os.chdir(captpath)
    #print(progpath4)

    captpath = os.path.join(progpath4, "data", "current_faces", "captured")
    writepath = os.path.join(progpath4, "data", "current_faces", "cropped")

    if os.path.exists(captpath) is False:
        os.makedirs(captpath)

    if os.path.exists(writepath) is False:
        os.makedirs(writepath)
    #print(captpath)
    for imagename in os.listdir(captpath):
        imagepath=os.path.join(captpath, imagename)
        image = cv2.imread(imagepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
        )
        #print("[INFO] Found {0} Faces.".format(len(faces)))
        os.chdir(writepath)
        for (x, y, w, h) in faces:
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            #print("[INFO] Object found. Saving locally.")
            status = cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)
            #print("[INFO] Image faces_detected.jpg written to filesystem: ", status)
#fc()


