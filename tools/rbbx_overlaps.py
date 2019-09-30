import numpy as np
import cv2


def rbbx_overlaps(boxes, query_boxes):
    '''
    Parameters
    ----------------
    boxes: (N, 5) --- x_ctr, y_ctr, height, width, angle
    query: (K, 5) --- x_ctr, y_ctr, height, width, angle   
    ----------------
    Returns
    ---------------- 
    Overlaps (N, K) IoU
    '''
           
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype = np.float32)
    
    for k in range(K):
        query_area = query_boxes[k, 2] * query_boxes[k, 3]
        for n in range(N):
            box_area = boxes[n, 2] * boxes[n, 3]
            #IoU of rotated rectangle
            #loading data anti to clock-wise
            rn = ((boxes[n, 0], boxes[n, 1]), (boxes[n, 3], boxes[n, 2]), -boxes[n, 4])
            rk = ((query_boxes[k, 0], query_boxes[k, 1]), (query_boxes[k, 3], query_boxes[k, 2]), -query_boxes[k, 4])
            int_pts = cv2.rotatedRectangleIntersection(rk, rn)[1]
            #print type(int_pts)		
            if  None != int_pts:
                order_pts = cv2.convexHull(int_pts, returnPoints = True)
                int_area = cv2.contourArea(order_pts)
                overlaps[n, k] = int_area * 1.0 / (query_area + box_area - int_area)
    return overlaps