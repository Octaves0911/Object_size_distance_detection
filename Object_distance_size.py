import numpy as np
import imutils
from imutils import contours
import cv2
import argparse
from scipy.spatial import distance as dst


def midpoint(ptsa,ptsb):
    return ((ptsa[0]+ptsb[0])*0.5,(ptsa[1]+ptsb[1])*0.5)

def ordered_points(pts):
    rect=np.zeros((4,2))
    sum=np.sum(pts,axis=1)
    rect[0]=pts[np.argmin(sum)]
    rect[2]=pts[np.argmax(sum)]
    diff=np.diff(pts,axis=1)
    rect[1]=pts[np.argmin(diff)]
    rect[3]=pts[np.argmax(diff)]

    return rect
def distance_from_camera(knownwidth,focal_length,perwidth):
    return (knownwidth*focal_length)/perwidth

arg=argparse.ArgumentParser()
arg.add_argument("-i","--rimage",required=True,help="The reference image")
arg.add_argument("-j","--image",required=True,help="The image file")
arg.add_argument("-w","--width",required=True,help="The width of reference object")
arg.add_argument("-d","--distance",required=True,help="The distance of the reference screen")

args=vars(arg.parse_args())

rimage=cv2.imread(args["rimage"])
image=cv2.imread(args["image"])



aspecti=image.shape[0]/600
rimage=imutils.resize(rimage,height=600)
image=imutils.resize(image,height=600)

##working on reference image

rgray=cv2.cvtColor(rimage,cv2.COLOR_BGR2GRAY)
rblurred=cv2.GaussianBlur(rgray,(7,7),0)
redged=cv2.Canny(rblurred,50,100)

pixelpermatric=None

rcnts=cv2.findContours(redged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
rcnts=rcnts[0]
rcnts=max(rcnts,key=cv2.contourArea)
##peri=cv2.arcLength(rcnts,closed=True)
##approx=cv2.approxPolyDP(rcnts,peri*0.1,closed=True)
##approx.reshape(4,2)

##rordpoints=ordered_points(approx)

known_distance=float(args["distance"])
known_width=float(args["width"])

brect=cv2.minAreaRect(rcnts)

focal_length=(brect[1][0]*known_distance)/known_width

## working on image
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred=cv2.GaussianBlur(gray,(5,5),0)
edged=cv2.Canny(blurred,50,100)
dilate=cv2.dilate(edged,None,iterations=1)
erode=cv2.erode(dilate,None,iterations=1)

cnts=cv2.findContours(erode.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=cnts[0]

(cnts,_)=contours.sort_contours(cnts)
output=image.copy()

for c in cnts:
    if cv2.contourArea(c)<100:
        continue
    
    box1=cv2.minAreaRect(c) ## it will  return the 
    box=cv2.cv.BoxPoints(box1) if imutils.is_cv2() else cv2.boxPoints(box1) ##boxpoint will provide the four coordinated of the box
    box=np.array(box,dtype='int')
    orderedbox=ordered_points(box)
    cv2.drawContours(output,[box.astype("int")],-1,(0,255,0),2)

    box=ordered_points(box)
    cv2.drawContours(output,[box.astype("int")],-1,(0,255,0),2)

    for(x,y) in orderedbox:
        cv2.circle(output,(int(x),int(y)),5,(0,0,255),-1)
        (tl,tr,br,bl)=orderedbox

        ## Now I have to get the midpoints of the four edges
        (tltrX,tltrY)=midpoint(tl,tr)
        (blbrX,blbrY)=midpoint(bl,br)
        (tlblX,tlblY)=midpoint(tl,bl)
        (trbrX,trbrY)=midpoint(tr,br)

        cv2.circle(output,(int(tltrX),int(tltrY)),5,(255,0,0),-1)
        cv2.circle(output,(int(blbrX),int(blbrY)),5,(255,0,0),-1)
        cv2.circle(output,(int(tlblX),int(tlblY)),5,(255,0,0),-1)
        cv2.circle(output,(int(trbrX),int(trbrY)),5,(255,0,0),-1)

        cv2.line(output,(int(tltrX),int(tltrY)),(int(blbrX),int(blbrY)),(255,0,255),2)
        cv2.line(output,(int(tlblX),int(tlblY)),(int(trbrX),int(trbrY)),(255,0,255),2)

        da=dst.euclidean((tltrX,tltrY),(blbrX,blbrY)) ## da and db will be in the form of pixels
        db=dst.euclidean((tlblX,tlblY),(trbrX,trbrY))

        if pixelpermatric is None:
            pixelpermatric=db/0.9555
            inches=distance_from_camera(known_width,focal_length,box1[1][0])
        ## compuing the actual distance
        dimA=da/pixelpermatric ## dima and dimb are in the form of inches
        dimB=db/pixelpermatric
        cv2.putText(output,"{:.1f}in".format(dimA),(int(tltrX-15),int(tltrY-10)),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,22,255),2)
        cv2.putText(output,"{:.1f}in".format(dimB),(int(trbrX-15),int(trbrY-10)),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
        cv2.putText(output, "Distance of screen from camera %.2fft" % (inches / 12),(output.shape[1] - 600, output.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 1)
cv2.imshow("output",output)
cv2.waitKey(0)
    












    
