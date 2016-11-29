import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def getBoundingROIfromQuad(quad):
    roi=(0,0,0,0)    
    roi[0]=min(quad[0])
    roi[1]=min(quad[1])
    roi[2]=max(quad[0])
    roi[3]=max(quad[1])
    return roi


def isInside(point,quad):
    topleft=quad[:,0]        
    topright=quad[:,1]
    bottomright=quad[:,2]
    bottomleft=quad[:,3]
    
    v1=topright-topleft
    v2=bottomright-topright
    v3=bottomleft-bottomright
    v4=topleft-bottomleft
    
    v1=v1/np.linalg.norm(v1)
    v2=v2/np.linalg.norm(v2)
    v3=v3/np.linalg.norm(v3)
    v4=v4/np.linalg.norm(v4)
    
    p1=point-topleft
    p2=point-topright
    p3=point-bottomright
    p4=point-bottomleft
    
    p1=p1/np.linalg.norm(p1)
    p2=p2/np.linalg.norm(p2)
    p3=p3/np.linalg.norm(p3)
    p4=p4/np.linalg.norm(p4)    
    
    val1=np.cross(v1,p1)
    val2=np.cross(v2,p2)
    val3=np.cross(v3,p3)
    val4=np.cross(v4,p4)
    if val1 > 0 and val2 > 0 and val3 > 0 and val4 > 0 :
        return True
    return False


    
    
def acquireTrackingQuad(frame):
    cv2.namedWindow("inputwindow")
    trackingQuad=np.zeros((2,4))
    n_quad=[0]
    def mousecallBack(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            if(n_quad[0]>0):
                cv2.line(frame, (x,y), (int(trackingQuad[0,max(n_quad[0]-1,0)]), int(trackingQuad[1, max(n_quad[0]-1,0)])), (255,0,0), 10)
            else:
                cv2.circle(frame,(x,y),10,(255,0,0),-1)
            if(n_quad[0]==3):
                cv2.line(frame,(x,y),(int(trackingQuad[0,0]),int(trackingQuad[1,0])),(255,0,0),10)
            trackingQuad[0,n_quad[0]]=x
            trackingQuad[1,n_quad[0]]=y
            n_quad[0]+=1
            
        return
    cv2.setMouseCallback("inputwindow",mousecallBack)
    while 1:
        cv2.imshow("inputwindow",frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    return trackingQuad
    

path=os.path.normpath('D:/Testvideos/video_rendered.mp4')

cap=cv2.VideoCapture(path)
if cap.isOpened():
    ret,frame=cap.read()

 
# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv2.SURF(4000)

# Find keypoints and descriptors directly

if ret:
    trackingQuad=acquireTrackingQuad(frame)
    
prevkp, prevdes = surf.detectAndCompute(frame,None)
prevFrame=frame

bf = cv2.BFMatcher()
while(1):
    #read and detect
    ret,frame=cap.read()
    kp, des = surf.detectAndCompute(frame,None)

    #match       
    matches = bf.match(prevdes,des)
    
    matches = sorted(matches, key = lambda x:x.distance)

    #visualize
    kp=[keypoint for keypoint in kp if isInside(keypoint.pt,trackingQuad)]
    visframe = cv2.drawKeypoints(frame,kp,None,(255,0,0),4)
    cv2.imshow("inputwindow",visframe)
        
    #iterate
    prevdes=des
    prevkp=kp
    prevFrame=frame
    
    cv2.waitKey(0)
    #if cv2.waitKey(20) & 0xFF == 27:
    #    break
cv2.destroyAllWindows()
    
