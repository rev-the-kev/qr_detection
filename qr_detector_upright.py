#!/usr/bin/env python
'''
This script performs three main tasks after loading an image. It first
detects a QR code within the image, then determines the lateral distance the code
is located from the camera. Finally, it decodes the QR code.
Written by Kevin Daniel, May 30, 2018. Code written using QR detection methods
implemented by Bharath Prabhuswamy.
https://github.com/bharathp666/opencv_qr
Assumption: reference code accounts for possibility of QR code not being aligned in the correct
orientation. For our particular application we know that the QR code will always be
upright, and thus we can reduce the complexity of the code.
'''
import cv2
import numpy as np
import cmath
'''
Given two points, this function returns the Euclidean distance
between the two.
'''
def getEuclideanDistance(P, Q):
    D = np.subtract(P, Q)
    return np.linalg.norm(D)
'''
This function takes in three points. Using vector norms and cross products,
it is able to calculate the shortest distance between the third point and
the line created by the first two. It returns this minimum distance.
'''
def distanceFromLine(L, M, J):
    a = -((M[1] - L[1]) / (M[0] - L[0]))
    b = 1.0
    c = ((M[1] - L[1]) / (M[0] - L[0])) * L[0]- L[1]
    pdist = (a * J[0] + (b * J[1]) + c) / np.sqrt((a * a) + (b*b))
    return pdist
'''
Two points are passed into this function. The function returns the slope
of the line created by the two points or zero if the two points are
vertically aligned. A flag is also returned. The flag is 0 if the line
is vertical, else it is 1.
'''
def getSlope(A, B):
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    if (dy != 0):
        m = dy/float(dx)
        return m
    else:
        return 0.0

def updateCorner(point, ref, current_max, current_vrtx):
    temp_dist = getEuclideanDistance(point, ref)

    if (temp_dist > current_max):
        return temp_dist, point
    else:
        return current_max, current_vrtx

def getVertices(contours, c_id, slope):
    box = cv2.boundingRect(contours[c_id])

    A = [box[0], box[1]]                    #Top left of bounding rectangle
    B = [box[0] + box[2], box[1]]           #Top right of bounding rectangle
    C = [box[0] + box[2], box[1] + box[3]]  #Bottom right of bounding rectangle
    D = [box[0], box[1] + box[3]]           #Bottom left of bounding rectangle

    W = [(A[0] + B[0]) / 2]
    W.append(A[1])
    X = [B[0]]
    X.append((B[1] + C[1]) / 2)
    Y = [(C[0] + D[0]) / 2]
    Y.append(C[1])
    Z = [D[0]]
    Z.append((D[1] + A[1]) / 2)

    max_dist = [0.0, 0.0, 0.0, 0.0]
    M0 = [0, 0]
    M1 = [0, 0]
    M2 = [0, 0]
    M3 = [0, 0]

    if (slope > 5 or slope < -5):
        for i in contours[c_id]:
            pd1 = distanceFromLine(C, A, i)
            pd2 = distanceFromLine(B, D, i)

            if (pd1 >= 0.0 and pd2 >= 0.0):
                max_dist[1], M1 = updateCorner(i, W, max_dist[1], M1)
            elif (pd1 > 0.0 and pd2 <= 0.0):
                max_dist[2], M2 = updateCorner(i, X, max_dist[2], M2)
            elif (pd1 <= 0.0 and pd2 < 0.0):
                max_dist[3], M3 = updateCorner(i, Y, max_dist[3], M3)
            elif (pd1 < 0.0 and pd2 >= 0.0):
                max_dist[0], M0 = updateCorner(i, Z, max_dist[0], M3)
    else:
        halfx = (A[0] + B[0]) / 2
        halfy = (A[1] + D[1]) / 2

        for i in contours[c_id]:
            if (i[0][0] < halfx and i[0][1] <= halfy):
                max_dist[2], M0 = updateCorner(i, C, max_dist[2], M0)
            elif (i[0][0] >= halfx and i[0][1] < halfy):
                max_dist[3], M1 = updateCorner(i, D, max_dist[3], M1)
            elif (i[0][0] > halfx and i[0][1] >= halfy):
                max_dist[0], M2 = updateCorner(i, A, max_dist[0], M2)
            elif (i[0][0] <= halfx and i[0][1] > halfy):
                max_dist[1], M3 = updateCorner(i, B, max_dist[1], M3)

    return [M0, M1, M2, M3]

def getIntersectionPoint(a1, a2, b1, b2):
    intersection = [0, 0]
    mb = (float(b2[1]) - b1[1])/(b2[0] - b1[0])
    bb = b1[1] - mb * b1[0]

    try:
        ma = (float(a2[1]) - a1[1])/(a2[0] - a1[0])
    except ZeroDivisionError:
        intersection[0] = a1[0]
    else:
        ba = a1[1] - ma * a1[0]
        intersection[0] = np.absolute((bb - ba)/(mb - ma))
    finally:
        intersection[1] = int(np.absolute(mb * intersection[0] + bb))
        intersection[0] = int(intersection[0])
        return intersection

def cross(v1, v2):
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return cross

def main():
    img = cv2.imread('qr_example.jpg', 1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (3,3), 0)
    #Implement adaptive thresholding in the future for robustness
    ret, thresh_img = cv2.threshold(blurred_img, 127, 255, cv2.THRESH_BINARY)
    canny_img = cv2.Canny(thresh_img, 100, 200)
    cont_img, contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mark = 0

    moments = []
    centroids = []
    for cont in contours:
        temp_moments = cv2.moments(cont)
        try:
            centroid_x = int(temp_moments['m10']/temp_moments['m00'])
            centroid_y = int(temp_moments['m01']/temp_moments['m00'])
        except ZeroDivisionError:
            x, y, w, h = cv2.boundingRect(cont)
            centroid_x = int(x + w/2)
            centroid_y = int(y + h/2)
        else:
            pass
        finally:
            centroids.append((centroid_x, centroid_y))
            moments.append(temp_moments)

    for x in range(len(contours)):
        approx_poly = cv2.approxPolyDP(contours[x], .02 * cv2.arcLength(contours[x], True), True)

        if (len(approx_poly) == 4):
            k = x
            c = 0

            while (hierarchy[0][k][2] != -1):
                k = hierarchy[0][k][2]
                c += 1

            if (c >= 5):
                if(mark == 0):
                    A = x
                elif(mark == 1):
                    B = x
                elif(mark == 2):
                    C = x
                mark += 1

    if (mark >= 3):
        AB = getEuclideanDistance(centroids[A], centroids[B])
        BC = getEuclideanDistance(centroids[B], centroids[C])
        CA = getEuclideanDistance(centroids[C], centroids[A])

        if (AB > BC and AB > CA):
            top = C
            if (centroids[A][0] < centroids[B][0]):
                bottom = A
                right = B
            else:
                bottom = B
                right = A
        elif (CA > AB and CA > BC):
            top = B
            if (centroids[A][0] < centroids[C][0]):
                bottom = A
                right = C
            else:
                bottom = C
                right = A
        else:
            top = A
            if (centroids[C][0] < centroids[B][0]):
                bottom = C
                right = B
            else:
                bottom = B
                right = C

        slope = getSlope(centroids[bottom], centroids[right])

        src = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype = "float32")
        dst = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]], dtype = "float32")

        if (top < len(contours) and right < len(contours) and bottom < len(contours) and cv2.contourArea(contours[top]) > 10 and cv2.contourArea(contours[right]) > 10 and cv2.contourArea(contours[bottom]) > 10):
            L = getVertices(contours, top, slope)
            M = getVertices(contours, right, slope)
            O = getVertices(contours, bottom, slope)
            N = getIntersectionPoint(M[1][0], M[2][0], O[3][0], O[2][0])

            src[0] = L[0][0]
            src[1] = M[1][0]
            src[2] = N
            src[3] = O[3][0]
            #begin debugging block
            ##################################################################
            cv2.circle(img, (M[1][0][0], M[1][0][1]), 2, (0,0,255), -1)
            cv2.circle(img, (M[2][0][0], M[2][0][1]), 2, (0,0,255), -1)
            cv2.circle(img, (O[3][0][0], O[3][0][1]), 2, (0,0,255), -1)
            cv2.circle(img, (O[2][0][0], O[2][0][1]), 2, (0,0,255), -1)
            cv2.circle(img, (src[0][0], src[0][1]), 2, (0,255,0), -1)
            cv2.circle(img, (src[1][0], src[1][1]), 2, (0,255,0), -1)
            cv2.circle(img, (src[2][0], src[2][1]), 2, (0,255,0), -1)
            cv2.circle(img, (src[3][0], src[3][1]), 2, (0,255,0), -1)
            ##################################################################
            #end debugging block
            cv2.imshow('Original Image', img)
            warp_matrix = cv2.getPerspectiveTransform(src, dst)
            qr_raw = cv2.warpPerspective(img, warp_matrix, (100, 100))
            cv2.imshow('QR Code', qr_raw)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
