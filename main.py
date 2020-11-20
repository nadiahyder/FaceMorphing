import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
import numpy.linalg as lin
import skimage.draw as draw
from skimage import color
import skimage as sk
import skimage.feature
import skimage.io as skio
import random
from scipy.spatial import Delaunay
import asf_read

import time
import imageio
import os
import re

POINT_COUNT = 68
FRAME_COUNT = 45

def weightsPerFrame():
	return np.linspace(0.0, 1.0, FRAME_COUNT)


def getCorrespondence(picA, picB):
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    corresp = np.zeros((68, 2))

    imgs = [picA, picB]
    list1 = []
    list2 = []
    j = 1

    for img in imgs:
        size = (img.shape[0], img.shape[1])
        if (j == 1):
            currList = list1
        else:
            currList = list2

        # get bounding boxes of face using face_detector, 1 means we upsample the picture once to make it bigger
        dets = face_detector(img, 1)

        j = j + 1

        for k, rect in enumerate(dets):

            # Get the landmarks/parts for the face in rect.
            shape = shape_predictor(img, rect)

            for i in range(0, 68):
                x = shape.part(i).x
                y = shape.part(i).y
                currList.append((x, y))
                corresp[i][0] += x
                corresp[i][1] += y

            # Add back the background
            currList.append((1, 1))
            currList.append((size[1] - 1, 1))
            currList.append(((size[1] - 1) // 2, 1))
            currList.append((1, size[0] - 1))
            currList.append((1, (size[0] - 1) // 2))
            currList.append(((size[1] - 1) // 2, size[0] - 1))
            currList.append((size[1] - 1, size[0] - 1))
            currList.append(((size[1] - 1), (size[0] - 1) // 2))

    # Add back the background
    narray = corresp / 2
    narray = np.append(narray, [[1, 1]], axis=0)
    narray = np.append(narray, [[size[1] - 1, 1]], axis=0)
    narray = np.append(narray, [[(size[1] - 1) // 2, 1]], axis=0)
    narray = np.append(narray, [[1, size[0] - 1]], axis=0)
    narray = np.append(narray, [[1, (size[0] - 1) // 2]], axis=0)
    narray = np.append(narray, [[(size[1] - 1) // 2, size[0] - 1]], axis=0)
    narray = np.append(narray, [[size[1] - 1, size[0] - 1]], axis=0)
    narray = np.append(narray, [[(size[1] - 1), (size[0] - 1) // 2]], axis=0)

    return [size, imgs[0], imgs[1], list1, list2, narray]

def showFeaturePoints(picA, picB, featuresA, featuresB):
    plt.scatter(*zip(*featuresA))
    plt.imshow(picA)
    plt.show()

    plt.scatter(*zip(*featuresB))
    plt.imshow(picB)
    plt.show()

def showDelaunay(pic, points, tri):
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.imshow(pic)
    plt.show()

## Get interpolated points
def avg_pair_points(pointsA, pointsB, alpha):
    #return alpha * np.array(pointsB) + (1-alpha) * np.array(pointsA)
    final_points = []
    for i in range(0, len(pointsA)):
        final_points.append((alpha * pointsA[i] + (1 - alpha) * pointsB[i]))
    return np.array(final_points)

## Get triangle coordinates in image
def triangle_bool_matrix(triangle, image_shape):
    tri_buf = triangle
    shape = (image_shape[1], image_shape[0], image_shape[2])
    points = draw.polygon(tri_buf[:,0], tri_buf[:,1], shape=shape)
    return np.vstack([points, np.ones(len(points[0]))])

## Compute Affine matrix
def affine(triangle, target):
    A = np.matrix([triangle[:,0], triangle[:, 1], [1, 1, 1]])
    B = np.matrix([target[:,0], target[:, 1], [1, 1, 1]])
    return B * lin.inv(A)


## Apply inverse transformation based on mask
def apply_masked_affine(mask, image, src_tri, target_tri):
    A = affine(src_tri, target_tri)

    # Invert the mask so it's
    # [x]            [y]
    # [y] instead of [x]
    # [1]            [1]
    final_mask = mask.copy()
    final_mask[0] = mask[0]
    final_mask[1] = mask[1]

    affined = (lin.inv(A) * final_mask).astype(np.int)
    cc, rr, stub = affined

    final_mask = final_mask.astype(np.int)
    canvas = np.zeros_like(image)
    canvas[final_mask[1], final_mask[0]] = image[rr, cc]
    return canvas

def blendPair(tri_list, images, points, alpha):
    avg = avg_pair_points(points[0], points[1], alpha)

    assert(images[0].shape == images[1].shape)

    morphed = np.zeros_like(images[0]).astype(np.float)

    # out1 and out2 are the images after midway calculation
    out1 = np.zeros_like(images[0]).astype(np.float)
    out2 = np.zeros_like(images[0]).astype(np.float)


    for tri in tri_list:
        target_tri = avg[tri]
        target_mask = triangle_bool_matrix(target_tri, images[0].shape)

        src_tri = points[0][tri]

        warp1 = alpha * apply_masked_affine(target_mask, images[0],
                                                src_tri, target_tri) / 255
        out1+=warp1

        morphed_tri = warp1

        src_tri = points[1][tri]

        warp2 = (1-alpha) * apply_masked_affine(target_mask, images[1],
                                                     src_tri, target_tri) / 255

        morphed_tri += warp2
        out2+=warp2
        morphed += morphed_tri

    # out1 and out2 are the warped faces
    # skio.imshow(out1)
    # skio.show()
    #
    # skio.imshow(out2)
    # skio.show()

    #skio.imsave("nadia_man_warped.png", out1)
    #skio.imsave("danes_warped.png", out2)

    morphed[morphed>1] = 1
    return morphed, avg

def process_points(points, shape):
    points.append((0, 0))
    points.append((0, shape[0]))
    points.append((shape[1], 0))
    points.append((shape[1], shape[0]))
    return np.array(points)

def findMidwayFace(imgA, imgB, pointsA, pointsB, alpha=0.5):
    shape = imgA.shape
    pointsA = process_points(pointsA, shape)
    pointsB = process_points(pointsB,shape)

    triA = Delaunay(pointsA)
    triangles = triA.simplices
    face, _ = blendPair(triangles, [imgA, imgB],
                         [pointsA, pointsB], alpha)
    return face
    #skio.imsave("n_man_morph.png", face)

def convertToGif(name, frames):
    imageio.mimsave(name, frames, format='GIF', duration=1/30)
    return

## Save movie based on two images
def morphSequence(imgA, imgB, pointsA, pointsB, numFrames):
    shape = imgA.shape
    pointsA = process_points(pointsA, shape)
    pointsB = process_points(pointsB, shape)

    triA = Delaunay(pointsA)
    triangles = triA.simplices
    mov = []
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    for i in range(0, numFrames + 1):
        alpha = float((numFrames - i) / numFrames)
        strt = time.time()
        paint, _ = blendPair(triangles, [imgA, imgB], [pointsA, pointsB], alpha)

        mov.append(paint)

    convertToGif("1-2.gif", mov)


def getAsfPoints(file_name):
    lines = asf_read.read_asf(file_name)
    points = []
    for line in lines:
        data = line.split(" \t")
        points.append((float(data[2]), float(data[3])))
    points.append((0., 0.))
    points.append((1., 0.))
    points.append((0., 1.))
    points.append((1., 1.))
    return np.array(points)

# regex = ".*-1f\.jpg"
def readDatabase(path, regex):
    file_ls = []
    points_ls = []
    for file in os.listdir(path):
       if re.match(regex, file):
           img = skio.imread(os.path.join(path, file))

           file_ls.append(img)
           asf_file = re.sub("jpg", "asf", os.path.join(path, file))
           points_ls.append(getAsfPoints(asf_file))

    shape = file_ls[0].shape
    for point_set in points_ls:
        for i in range(0, len(point_set)):
            point_set[i] = (point_set[i][0]*shape[1], point_set[i][1]*shape[0])

    return file_ls, points_ls

# get average of points from population
def avg_population_points(pop_points):
    avg = pop_points[0]
    for i in range(1, len(pop_points)):
        avg = np.add(avg, pop_points[i])
    return avg/len(pop_points)

def populationAverage(tri_list, images, point_sets):
    avg = avg_population_points(point_sets)
    pop_mean = np.zeros_like(images[0]).astype(np.float)
    out1 = np.zeros_like(images[0]).astype(np.float)

    #skio.imsave("dane1.png",images[1])

    for tri in tri_list:
        target_tri = avg[tri]
        target_mask = triangle_bool_matrix(target_tri, images[0].shape)
        for i in range(0, len(images)):
            src_tri = np.array(point_sets[i])[tri]
            warp = (1/len(images)) * apply_masked_affine(target_mask, images[i],
                                                              src_tri, target_tri) / 255
            final_tri = warp

            if(i == 1):
                out1 +=warp

            pop_mean += final_tri
    #skio.imsave("dane1_warped.png",out1)
    #skio.show()
    pop_mean[pop_mean > 1] = 1
    return pop_mean, avg


## Save population mean face
def computeMeanFace(path, regex, dest_img):
    images, points = readDatabase(path, regex)
    triA = Delaunay(points[0])
    triangles = triA.simplices
    final, avg_points = populationAverage(triangles, images, points)
    skio.imsave(dest_img, final)
    skio.imshow(final)
    skio.show()
    return final


def showTriangulation(picA, picB, featuresA, featuresB):
    pointsA = np.array(featuresA)
    triA = Delaunay(pointsA)
    showDelaunay(picA, pointsA, triA)

    pointsB = np.array(featuresB)
    triB = Delaunay(pointsB)
    showDelaunay(picB, pointsB, triB)

    avg = np.dot(pointsA + pointsB, 0.5)
    triAvg = Delaunay(avg)
    showDelaunay(picA, avg, triAvg)
    showDelaunay(picB, avg, triAvg)

def show(nameA, nameB, listA, listB):
    skPicA = skio.imread(nameA)
    skPicB = skio.imread(nameB)
    showFeaturePoints(skPicA, skPicB, listA, listB)
    showTriangulation(skPicA, skPicB, listA, listB)


def morphFiles(nameA, nameB):
    picA = cv2.imread(nameA)
    picB = cv2.imread(nameB)

    skPicA = skio.imread(nameA)
    skPicB = skio.imread(nameB)

    [size, img1, img2, pointsA, pointsB, list3] = getCorrespondence(picA, picB)

    face = findMidwayFace(skPicA, skPicB, pointsA, pointsB, 0.5)
    skio.imshow(face)
    skio.show()
    morphSequence(skPicA, skPicB, pointsA, pointsB, 45)

def performMeanFace(nameA, nameB):
    picA = cv2.imread(nameA)
    picB = cv2.imread(nameB)

    skPicA = skio.imread(nameA)
    skPicB = skio.imread(nameB)

    [size, img1, img2, pointsA, pointsB, list3] = getCorrespondence(picA, picB)

    face = findMidwayFace(skPicA, skPicB, pointsA, pointsB, 0.5)
    skio.imshow(face)


def findCaricature(nameA, nameB, alpha = 1.5):

    picA = skio.imread(nameA)
    picB = skio.imread(nameB)

    [size, img1, img2, pointsA, pointsB, list3] = getCorrespondence(picA, picB)
    face = findMidwayFace(picA, picB, pointsA, pointsB, 1.5)
    skio.imshow(face)
    skio.show()

    skio.imsave("caricature0.1.png", face)

def main():

    morph = False
    meanFace = False
    caricature = False
    changeToMale = False
    classMorph = False

    if morph:
        nameA = "nadia.jpg"
        nameB = "george.jpg"
        morphFiles(nameA, nameB)

    if meanFace:
        popMean = computeMeanFace("imm_face_db/", ".*-1f\.jpg", "danes_mean.png")
        nameA = "nadia2.jpg"
        nameB = "danes.png"
        performMeanFace(nameA, nameB)

    if caricature:
        nameA = "nadia2.jpg"
        nameB = "danes.png"
        findCaricature(nameA, nameB,1.2)

    if changeToMale:
        nameA = "nadia3.jpg"
        nameB = "indian.png"
        performMeanFace(nameA, nameB)

    if classMorph:
        nameA = "image1.png"
        nameB = "image2.jpg"
        morphFiles(nameA, nameB)

if __name__ == "__main__":
    main()
