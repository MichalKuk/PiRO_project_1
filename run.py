from scipy import ndimage
from skimage import io
from skimage.transform import resize
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys

"""
Program będzie wywoływany z linii komend z dwoma parametrami:
1. Ścieżką do katalogu z plikami – path
Może to być ścieżka względna lub bezwzględna.
Do operacji na ścieżkach proszę używać biblioteki os.path lub pathlib a nie zwykłych operacji na stringachn 
które w bardzo wielu wypadkach powodowały problemy z działaniem programów!
2. Liczbą obrazków do wczytania – N
Uwaga: w katalogu może znajdować się więcej obrazków – należy je zignorować.

W podanym katalogu path będą znajdowały się pliki o nazwach k.png gdzie k jest liczbą z zakresu 0…N-1.

Wyjście:
Na standardowe wyjście program powinien wypisać N linii. W i-tej linii (licząc od zera: 0..N-1) 
powinna znaleźć się przynajmniej jedna liczba mówiąca z którym elementem należy sparować i-ty obrazek.
"""

def check_orient(image):
    height, width = image.shape
    # width, height = image.shape
    divider = 2
    black_pxls = []
    black_pxls.append(np.sum(image[:height // divider, :] == 0))  # up
    black_pxls.append(np.sum(image[(divider - 1) * (height // divider):, :] == 0))  # down    
    black_pxls.append(np.sum(image[:, :width // divider] == 0))  # left
    black_pxls.append(np.sum(image[:, (divider - 1) * (width // divider):] == 0))  # right
    
    side = black_pxls.index(max(black_pxls))
    if side == 0:
        return cv2.rotate(image, cv2.ROTATE_180)
    if side == 2:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if side == 3:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    return image


def cut_background(image):
    
    upIndex = 0
    leftIndex = 0
    rightIndex = 0
    downIndex = 0
    
    for ind, row in enumerate(image):
        if any(row) > 0:
            upIndex = ind
            break
            
    for ind, column in enumerate(image.T):
        if any(column) > 0:
            leftIndex = ind
            break  
            
    for ind, row in enumerate(np.flip(image)):
        if any(row) > 0:
            downIndex = ind
            break
            
    for ind, column in enumerate(np.flip(image.T)):
        if any(column) > 0:
            rightIndex = ind
            break     
            
    newImage = image[upIndex:image.shape[0] - downIndex]
    
    newImage = np.delete(newImage, np.s_[newImage.shape[1] - rightIndex:newImage.shape[1]], 1) #column
    newImage = np.delete(newImage, np.s_[0:leftIndex], 1)

    return newImage


def cut_white_block(image):

    downIndex = 0
    height, width = image.shape
    accuracy = 0.02

    for ind, row in enumerate(np.flip(image)):
        left = int(width*accuracy)
        right = int(width-(width*accuracy))
        if all(row[left:right]) > 0: # wszystkie pixele w wierszu nie są czarne (są białe)
            downIndex = ind
            break

    newImage = image[height - downIndex:]
    
    return newImage


def calculate_similarity(image1, image2):
    number_of_equal_elements = np.sum(image1==image2)
    total_elements = image1.shape[0] * image1.shape[1]
    percentage = number_of_equal_elements/total_elements
    
    return percentage

def simple_find_contours(images):
    pixelContours = []
    neighborhood = [(-1,-1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for image in images:
        pixC = []
        for row, line in enumerate(image):
            for column, pixel in enumerate(line):
                if(pixel == 0): # jeśli tło (czarne), to opuść tę iterację
                    continue
                for x,y in neighborhood: # jeśli pixel figury (biały), to przejrzyj jego siąsiadów
                    try:
                        if image[row + x, column + y] == 0: # if sąsiad pixela jest tłem (czarny), to pixel (jego współrzędne) dodaj do listy krawędzi
                            pixC.append((row, column))
                            break
                    except:
                        print("err")
        pixelContours.append(pixC) # dodaj cały kontur (ciąg pixeli) danego obrazka do listy konturów
    return pixelContours


def simple_find_contours(image):
    neighborhood = [(-1,-1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    pixC = []
    for row, line in enumerate(image):
        for column, pixel in enumerate(line):
            if(pixel == 0): # jeśli tło (czarne), to opuść tę iterację
                continue
            for x,y in neighborhood: # jeśli pixel figury (biały), to przejrzyj jego siąsiadów
                try:
                    if image[row + x, column + y] == 0: # if sąsiad pixela jest tłem (czarny), to pixel (jego współrzędne) dodaj do listy krawędzi
                        pixC.append((row, column))
                        break
                except:
                    print("err")
    return pixC

def hough_rotate_image(image, contursMap):
    angles = []
    lines = cv2.HoughLinesP(contursMap, 1, math.pi / 180.0, 50, minLineLength=50, maxLineGap=5)
    for x1, y1, x2, y2 in lines[0]:
        #cv2.line(contursMap, (x1, y1), (x2, y2), (255, 0, 0), 3) <- not needed
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
        
    median_angle = np.median(angles)
    # RYSOWANIE
    # axs[math.floor(index / 2),index % 2].imshow(imgs_rotated[-1], cmap='gray', vmin=0, vmax=255) 
    # index += 1
    return ndimage.rotate(image, median_angle)

if __name__ == "__main__":
    images = []

    for i in range(int(sys.argv[2])): 
        # path = Path(str(sys.argv[1])+'\\'+str(i)+'.png)
        p = os.path.abspath(sys.argv[1])
        path = os.path.join(p, str(i)+'.png')
        image = io.imread(path, as_gray=True)
        images.append((int(i), image)) 

    images.sort(key=lambda x: x[0])
    images = [img for _, img in images] # zostawia images bez indeksów

    # RYSOWANIE
    # fig, axs = plt.subplots(math.ceil(len(images)/2),2, figsize=(12,12))
    # for index, image in enumerate(images):
    #     axs[math.floor(index / 2), index % 2].imshow(image, cmap='gray', vmin=0, vmax=255) 
    # plt.show()

    contours = []
    contursMaps = []
    for image in images:     
        contursMaps.append(np.zeros(image.shape, dtype="uint8"))

    pixelContours = []

    for image in images:
        try:
            pixelContours.append(simple_find_contours(image)) # dodaj cały kontur (ciąg pixeli) danego obrazka do listy konturów
        except:
            pixelContours.append(image)
      
    for img_id, contour in enumerate(pixelContours): #kontur jednego obrazka
        for contour_pixel in contour: #jeden pixel (x,y) w konturze
            contursMaps[img_id][contour_pixel[0]][contour_pixel[1]] = 1 # zamaluj na biało pixel odpowiadający pixelowi konturu


    # RYSOWANIE
    # fig, axs = plt.subplots(int(len(images)/2),2, figsize=(12,12))
    # index = 0
    # for image in contursMaps:
    #     axs[math.floor(index / 2),index % 2].imshow(image, cmap='gray', vmin=0, vmax=1) 
    #     index += 1
    # plt.show()


    # RYSOWANIE
    # fig, axs = plt.subplots(int(len(images)/2),2, figsize=(12,12))
    imgs_rotated = []
    for image, contursMap in zip(images, contursMaps):
        try:
            imgs_rotated.append(hough_rotate_image(image, contursMap))
        except:
            imgs_rotated.append(image)

    del contursMaps
    del images


    removed_background = []
    for image in imgs_rotated:
        try:
            removed_background.append(cut_background(image))
        except:
            removed_background.append(image)
    del imgs_rotated


    imgs_fliped = [] 
    for image in removed_background:
        try:
            imgs_fliped.append(check_orient(image))  
        except:
            imgs_fliped.append(image) 
    del removed_background

    ready_images = []
    for image in imgs_fliped:
        try:
            ready_images.append(cut_white_block(image))
        except:
            ready_images.append(image)

    # RYSOWANIE
    # fig, axs = plt.subplots(int(len(images)/2),2, figsize=(12,12))
    # index = 0
    # for image in ready_images:
    #     axs[math.floor(index / 2),index % 2].imshow(image, cmap='gray', vmin=0, vmax=1) 
    #     index += 1
    # plt.show()

    minHeight, minWidth = ready_images[0].shape
    for image in ready_images:
        h,w = image.shape
        if h < minHeight:
            minHeight = h 
        if w < minWidth:
            minWidth = w

    if(minHeight * 6 < minWidth):
        minHeight = int(minWidth / 6) #Zapobiega przeskalowaniu wszystkich do bardzo małego prostokąta jeżeli któraś figura była prostokątem i usuwanie białego bloku wycięło praktycznie całość
    resized_images = []
    for index, image in enumerate(ready_images):
        h,w = image.shape
        
        if h >= minHeight and w >= minWidth:
            resized_images.append(cv2.resize(image, dsize=(minWidth, minHeight), interpolation=cv2.INTER_CUBIC))
        else:
            zeros = np.zeros(imgs_fliped[index].shape, dtype="uint8")
            concatenated = np.concatenate((imgs_fliped[index], zeros))
            resized_images.append(cv2.resize(concatenated, dsize=(minWidth, minHeight), interpolation=cv2.INTER_CUBIC))

    del ready_images
    del imgs_fliped
    # RYSOWANIE
    # fig, axs = plt.subplots(int(len(resized_images)/2),2, figsize=(12,12))
    # index = 0
    # for image in resized_images:
    #     axs[math.floor(index / 2),index % 2].imshow(image, cmap='gray', vmin=0, vmax=255) 
    #     index += 1 
    # plt.show()


    # output_file = open("output.txt","w")

    for index, image in enumerate(resized_images):
        similars = []
        for index2, image2 in enumerate(resized_images):
            if index == index2:
                continue
            sim = calculate_similarity(np.flip(255 - image),image2)
            similars.append((index2, sim))
                        
        similars.sort(key=lambda x: x[1], reverse=True)

        string_for_file = str(index)+" :"
        string = ""
        for i in similars:
            string += " "+str(i[0])
            string_for_file += " "+str(i[0])
        print(string)
        # print(string_for_file, file=output_file)

    # output_file.close()