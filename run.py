from scipy import misc
import glob
from skimage import io
from skimage.transform import resize
import math
import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
import imutils
import cv2
from skimage.measure import find_contours
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






images = []

for i in range(sys.argv[2]): 
    # path = Path(str(sys.argv[1])+'\\'+str(i)+'.png)
    path = os.path.join(str(sys.argv[1]), str(i), '.png')
    image = io.imread(path, as_gray=True)
    images.append((int(i), image)) 

images.sort(key=lambda x: x[0])
images = [img for _, img in images] # zostawia images bez indeksów

# RYSOWANIE
# fig, axs = plt.subplots(int(len(images)/2),2, figsize=(12,12))
# for index, image in enumerate(images):
#     axs[math.floor(index / 2),index % 2].imshow(image, cmap='gray', vmin=0, vmax=255) 

# numpyImages = []
# for image in images:
#     numpyImages.append(np.array(image))


# contours = []
# for image in images:    
#     contours.append(find_contours(image,0.5))

# contours_floors = []
# for cont in contours:
#     contours_floors.append([(r,c) for r,c in np.floor(cont[0])])

contours = []
for image in images:    
    contours.append([(r,c) for r,c in np.floor(find_contours(image,0.5)[0])]) 

contursMaps = []
for image in images:
    contursMaps.append(np.zeros(image.shape, dtype="uint8"))


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


# for index, image in enumerate(images):
#     for r, line in enumerate(contursMaps[index]):
#         for c, _ in enumerate(line):
#             if((r,c) in pixelContours[index]):
#                 contursMaps[index][r,c] = 1
                
for img_id, contour in enumerate(pixelContours): #kontur jednego obrazka
    for contour_pixel in contour: #jeden pixel (x,y) w konturze
        contursMaps[img_id][contour_pixel[0]][contour_pixel[1]] = 1 # zamaluj na biało pixel odpowiadający pixelowi konturu

# RYSOWANIE
# fig, axs = plt.subplots(int(len(images)/2),2, figsize=(12,12))
# index = 0
# for image in contursMaps:
#     axs[math.floor(index / 2),index % 2].imshow(image, cmap='gray', vmin=0, vmax=1) 
#     index += 1


# RYSOWANIE
# fig, axs = plt.subplots(int(len(images)/2),2, figsize=(12,12))
from scipy import ndimage
# index = 0
imgs_rotated = []

for image, contursMap in zip(images, contursMaps):
    
    angles = []
    
    lines = cv2.HoughLinesP(contursMap, 1, math.pi / 180.0, 50, minLineLength=50, maxLineGap=5)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(contursMap, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
        
    median_angle = np.median(angles)
    imgs_rotated.append(ndimage.rotate(image, median_angle))
    # RYSOWANIE
    # axs[math.floor(index / 2),index % 2].imshow(imgs_rotated[-1], cmap='gray', vmin=0, vmax=255) 
    # index += 1



# fig, axs = plt.subplots(int(len(images)/2),2, figsize=(12,12))
# index = 0

imgs_fliped = []

for image in imgs_rotated:
    imgs_fliped.append(check_orient(image))   
    # axs[math.floor(index / 2),index % 2].imshow(imgs_fliped[-1], cmap='gray', vmin=0, vmax=255) 
    # index += 1 