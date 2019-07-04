"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np

#from matplotlib import pyplot as plt
def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    (x0, y0) = p0
    (x1, y1) = p1
    
    distance = ((x0-x1)**2 + (y0 - y1)**2)**0.5
    return distance


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    
    
    #print ('image shape in corner =' + str(np.shape(image)))

    #n,m,p = np.shape(image)
    
     
    corners_final = [(0,0), (0,np.shape(image)[0] - 1), (np.shape(image)[1] - 1,0),(np.shape(image)[1] - 1,np.shape(image)[0] - 1)]
    #corners_final = [(0,0), (0,n-1), (m-1,0), (m-1,n-1)]
    #print corners_final
    return corners_final


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """

    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    bilateral_filtered_image = cv2.bilateralFilter(src=image, d=10, sigmaColor=200, sigmaSpace=300)
    gray = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    #print np.shape(img)
    #blur = cv2.bilateralFilter(img,9,75,75)
    corners = cv2.cornerHarris(gray,9,11,0.04)
    
    
    #dst = cv2.dilate(dst,None)
    
    (corners_new_x, corners_new_y)  = np.where(corners>0.055*corners.max())
    corners2 = np.column_stack((corners_new_x, corners_new_y))
    #print len(corners_new_x)
    #corners_new = corners[corners > 0.01 * corners.max()]
    
#    for i in range(len(corners_new_x)):
#        for j in range(len(corners_new_x)):
#            corners = corners +  (corners_new_x, corners_new_y)
    
    #img[dst>0.01*dst.max()]=[0]
    corners2 = np.float32(corners2)
    
#    Z = edges.reshape((-1,1))
#    Z = np.float32(Z)
#    print np.shape(Z)
    #temp, classified_points, center = cv2.kmeans(data=np.asarray(edges), K=4, bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), attempts=10, flags=cv2.KMEANS_USE_INITIAL_LABELS)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    ret,label,center=cv2.kmeans(corners2,4,criteria,30,cv2.KMEANS_RANDOM_CENTERS)
    #A = dst[label.ravel()==0]
    #B = dst[label.ravel()==1]
    #(x[0],y[0]) = np.max(center[0,:].ravel())
    #center = np.uint8(center)    
    #image[tuple(center.T)] = [255,255,255]
    #cv2.imshow('image centers', image)
    #cv2.waitKey(0)
    center = center.astype('int')    
    center = center[center[:,1].argsort()]
    center[0:2,:] = center[0:2,:][center[0:2,0].argsort()]
    center[2:4,:] = center[2:4,:][center[2:4,0].argsort()]
    center[:,[0,1]] = center[:,[1,0]]    
    # print(center)

    # image[tuple(center.T)] = [0,0,255]
    # cv2.imshow('res2',image)

    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()
    
    center_list = [tuple(center[0,:]), tuple(center[1,:]), tuple(center[2,:]), tuple(center[3,:])]
    print(center_list)
    return center_list

    

def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    
    #print ('markers in image' + str(markers))
    cv2.line(image,markers[0],markers[1],(0,255,0),thickness)
    cv2.line(image,markers[1],markers[3],(0,255,0),thickness)
    cv2.line(image,markers[3],markers[2],(0,255,0),thickness)
    cv2.line(image,markers[2],markers[0],(0,255,0),thickness)
    
    
    return image


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    imageB_x,imageB_y  = np.meshgrid(np.arange(imageB.shape[1]), np.arange(imageB.shape[0]))
    imageB_xy = np.vstack([imageB_x.reshape(-1), imageB_y.reshape(-1)])
    # print(imageB_xy.shape)
    imageB_xy = np.vstack([imageB_xy, np.ones((1,imageB_xy.shape[1])).astype('int')])
    # print(imageA_xy.shape)
    
    h_inv = np.linalg.inv(homography)
    # print(h_inv)
    imageA_xy = np.matmul(h_inv, imageB_xy)
    # print(imageA_xy.shape)
    imageA_xy = imageA_xy/imageA_xy[2] 
    # print(imageA_xy.shape)
    
    
    # print(imageA_xy.max())
    # print(imageA_xy.min())
    
    col_1 = np.all([imageA_xy[0,:] >= 0, imageA_xy[0,:] < imageA.shape[1]],axis=0)    
    col_2 = np.all([imageA_xy[1,:] >= 0, imageA_xy[1,:] < imageA.shape[0]],axis=0)    
    
    cols = np.logical_and(col_1,col_2)        
    imageA_xy = imageA_xy[:,cols]
    
    # print(imageA_xy.shape)
    imageB_xy = imageB_xy[:,cols]
    
    # print(imageB_xy.shape)
    imageB_row = imageB_xy[1,:].astype('int')
    #   print len(imageB_rows)
    imageB_col = imageB_xy[0,:].astype('int')
    #   print len(imageB_cols)
    imageA_row = imageA_xy[1,:].astype('int')
    #   print len(imageA_rows)
    imageA_col = imageA_xy[0,:].astype('int')
    #   print len(imageA_cols)
    
    
    imageB[imageB_row,imageB_col] = imageA[imageA_row,imageA_col]

    #cv2.imshow('result',imageB)
    #cv2.waitKey(0)
    
        
    return imageB

def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    
    P = np.zeros((8,9),dtype=np.int)
    
    for i in range(0,4):
        P[2*i,:] = np.array([-1* src_points[i][0], -1 * src_points[i][1], -1, 0, 0, 0, src_points[i][0] * dst_points[i][0], src_points[i][1] * dst_points[i][0], dst_points[i][0]])
        P[2*i + 1,:] = np.array([0, 0, 0, -1 * src_points[i][0], -1 * src_points[i][1], -1, src_points[i][0] * dst_points[i][1], src_points[i][1] * dst_points[i][1], dst_points[i][1]])
    
    
    u,s,Vt = np.linalg.svd(P, full_matrices=True)
    V = np.transpose(Vt)
    
    h = V[:,8].reshape(3,3)
    h = h/h[2,2]
    
    
    return h

    

def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    
    video.release()
    
    yield None
    
    
#    