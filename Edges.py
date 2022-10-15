# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 22:32:40 2020

@author: user1
"""

import cv2 
import numpy as np 
import math  
import matplotlib.pyplot as plt 
import pandas as pd
import sys
import os
import csv
import gc



def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

"""
===========
    ===========
        ===========
            ===========
            Import and update the variables and create the directories.
            ===========
        ===========
    ===========
===========
"""

"""
=== CREATE THE DIRECTORY TO SAVE THE IMAGES
"""
if not os.path.isdir("\img_export"):
    os.mkdir("\img_export")


"""
===========
    ===========
        ===========
            ===========
            Analyze the images.
            ===========
        ===========
    ===========
===========
"""

linearSpace = [int(i) for i in linearSpace]
for i in linearSpace:
    fig = plt.figure(figsize=(16,9))
    
    img = cv2.imread(DIRECTORY + 'IMG_FRAMES/frame_'+str(i)+'.jpg', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    canny = (auto_canny(img))
    
    #edges = feature.canny(img, sigma=1)
    
    detected_circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 2*rMax, param1 = 1000, 
                                        param2 = 10, minRadius = rMin, maxRadius = rMax)
    
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for circle in detected_circles[0, :]: 
            x, y, r = circle[0], circle[1], circle[2]
            cv2.circle(img, (x, y), r, (0, 255, 0), 4) 
            plt.text(x, y, str(r), color='b', fontsize=36)
    
    cv2.rectangle(img, tuple(squares["LU"]), tuple(squares["LD"]), (255,0,0), 5)
    cv2.rectangle(img, tuple(squares["RU"]), tuple(squares["RD"]), (255,0,0), 5)
    
    plt.grid(linewidth=2)
    plt.imshow(img)
    plt.savefig(DIRECTORY + 'IMG_DISTANCE/_'+str(i)+'_radius.jpg', dpi = 120)
    plt.close("all")

    del img, detected_circles, fig
del x, y, r
      
"""
===========
    ===========
        ===========
            ===========
            Image prepare.
            ===========
        ===========
    ===========
===========
"""
gc.collect()
if CONFIG['prepare_images']:
    next_frame_to_prepare = 0
    if CONFIG['continue_preparation']:
        while True:
            next_frame_to_prepare += 1
            if not os.path.isfile(currentDir + '\\' + videoName + '\\IMG_DISTANCE\\frame_' + str(next_frame_to_prepare) + '.jpg'):
                break
    
        df = pd.read_csv(CSV_FILE, names=['Frame', 'Dist', 'Arduino', 'Temperature'])
        df['MA10'] = df['Dist'].rolling(window=10).mean()
        df['MA30'] = df['Dist'].rolling(window=30).mean()
        df['Temp10'] = df['Temperature'].rolling(window=10).mean()
        df['Temp30'] = df['Temperature'].rolling(window=30).mean()
        
    else:
        # DataFrame
        df = pd.read_csv(CSV_FILE, names=['Frame', 'Dist', 'Arduino', 'Temperature', 'MA10', 'MA30', 'Temp10', 'Temp30'])
        
    print(next_frame_to_prepare)    

    # Define the mininum and maximum value and the number of divisions to print
    minValue = 1
    maxValue = count_frames
    nDivision = 100
    
    # Get an Array with 100 values between minValue and maxValue and convert to int
    linearSpace = np.around(np.linspace(minValue, maxValue, nDivision), 0).tolist()
    linearSpace = [int(i) for i in linearSpace]
    
    print(" " + "-"*100)
    msg = "Realizando a análise de cada Frame..."
    print("|" + msg + " "*(100 - len(msg)) + "|")
    
    # Main loop, analyze all the images
    distance = 0
    for num in range(next_frame_to_prepare, count_frames+1):
        if (num in linearSpace):
            # Using the index you know how far you are in the progress bar
            index = linearSpace.index(num) + 1
            blankSpaces = (nDivision - index)
            percentage = str(round((100*(num+1))/(maxValue/minValue), 2))
            
            # Print '=' index times, and ' ' blankSpace times.
            sys.stdout.write("\r[" + "=" * index +  " " * blankSpaces + "] " + percentage + "%   ")
            sys.stdout.flush()
            
            # Garbage collector
            gc.collect()
            
        # Read image. 
        imgOriginal = cv2.imread(DIRECTORY + 'IMG_FRAMES/frame_' + str(num) + '.jpg', cv2.IMREAD_UNCHANGED)
        imgWithFigures = cv2.imread(DIRECTORY + 'IMG_FRAMES/frame_' + str(num) + '.jpg', cv2.IMREAD_UNCHANGED)
        
        # Convert to gray and creates a new image with a adaptative Threshold
        imgThreshold = cv2.adaptiveThreshold(cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY),255,\
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)
        
        imgCanny = cv2.bitwise_not(cv2.Canny(imgThreshold,200,1000))
        
        # Draw the rectangles
        cv2.rectangle(imgWithFigures, tuple(squares["LU"]), tuple(squares["LD"]), (0, 0, 255), 5)
        cv2.rectangle(imgWithFigures, tuple(squares["RU"]), tuple(squares["RD"]), (0, 0, 255), 5)
        
        # Detect the circles        
        detected_circles = cv2.HoughCircles(imgCanny, cv2.HOUGH_GRADIENT, 1, 2*rMax, param1 = 50, 
                                            param2 = 10, minRadius = rMin, maxRadius = rMax)
        
        # If there is a circle in the image
        if detected_circles is not None:
            # Convert the circle parameters a, b and r to integers. 
            detected_circles = np.uint32(np.around(detected_circles))
          
            x1 = 0; x2 = 0; y1 = 0; y2 = 0;
            for circle in detected_circles[0, :]: 
                x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
                draw = False
                
                # If the center of the circle is inside of the square, save the coordenates
                if ((x > squares['LU'][0] and x < squares['LD'][0]) and (y > squares['LU'][1] and y < squares['LD'][1])):
                    x1 = x
                    y1 = y
                    draw = True
                    
                # If the center of the circle is inside of the square, save the coordenates
                if ((x > squares['RU'][0] and x < squares['RD'][0]) and (y > squares['RU'][1] and y < squares['RD'][1])):
                    x2 = x
                    y2 = y
                    draw = True
                
                if draw:
                    # Draw the circumference of the circle. 
                    cv2.circle(imgWithFigures, (x, y), r, (0, 255, 0), 4) 
                    cv2.circle(imgWithFigures, (x, y), r, (0, 255, 0), 4) 
             
                    # Draw a small circle (of radius 1) to show the center. 
                    cv2.circle(imgWithFigures, (x, y), 3, (0, 255, 0), 5) 
                    cv2.circle(imgWithFigures, (x, y), 3, (0, 255, 0), 5) 
                
            # Calculate the distance
            if (x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0):
                cv2.line(imgWithFigures, (x1, y1), (x2, y2), (255, 0, 0), 4)
                distance = round(math.sqrt(abs(x2-x1)**2 + abs(y2-y1)**2), 2)
        
        fig = plt.figure(figsize=(16,9))
        gs = fig.add_gridspec(3, 3)
        
        # Plot the original image above
        fig.add_subplot(gs[0, 0]),plt.imshow(imgOriginal)
        plt.xticks([]), plt.yticks([])
        
        # Plot the image with circle, square and line in the middle
        fig.add_subplot(gs[1, 0]),plt.imshow(imgWithFigures)
        plt.xticks([]),plt.yticks([])
        
        # Plot the binary image below
        fig.add_subplot(gs[2, 0]),plt.imshow(cv2.cvtColor(imgCanny, cv2.COLOR_GRAY2RGB))
        plt.xticks([]),plt.yticks([])
        
        # Plot the graphic
        fig.add_subplot(gs[0:, 1:]),
        plt.ylabel('Distance (mm)',fontsize=14)
        
        if distance > minDistance:
            if CONFIG['dilatometry']:
                temperature = read_list[num][READ_INT_TEMP]
            else:
                temperature = read_list[frame_arduino[num]][READ_INT_TEMP]
            
            if len(df['Dist'].tail(9)) < 9:
                ma10 = float("NaN")
                temp10 = float("NaN")
            else:
                ma10 = (df['Dist'].tail(9).sum() + distance)/10
                temp10 = (df['Temperature'].tail(9).sum() + temperature)/10
            if len(df['Dist'].tail(30)) < 30:
                ma30 = float("NaN")
                temp30 = float("NaN")
            else:
                ma30 = (df['Dist'].tail(29).sum() + distance)/30
                temp30 = (df['Temperature'].tail(29).sum() + temperature)/30
            
            df = df.append({'Frame':num, 'Dist':distance, 'MA10':ma10, 'MA30':ma30, 'Temperature':temperature, 'Temp10':temp10, 'Temp30':temp30}, ignore_index=True)
        
        # If the X axis is Temperature, put the temperature values, else, the count value
        if CONFIG['graphic_in_temperature']:
            plt.xlabel('Temperature (ºC)',fontsize=14)
            if CONFIG['dilatometry']:
                plt.plot(df['Temperature'], (21*df['Dist']/df['Dist'].head(1).iloc[0]), label="Frame distance")
            #plt.plot(df['Temp10'], df['MA10'], label="Moving Average 0.333 sec")
            else:
                plt.plot(df['Temp30'], df['MA30'], label="Moving Average 0.5 sec")
            
        else:
            plt.xlabel('Image number',fontsize=14)
            #plt.plot(df['Dist'],  label="Frame distance")
            plt.plot(df['MA10'],  label="Moving Average 0.333 sec")
            plt.plot(df['MA30'],  label="Moving Average 1 sec")
        
        if (distance != 0):
            plt.title("Distance = " + str(distance), fontsize=20)
        else:
            plt.title("Distance = 000.00", fontsize=20)
    
        plt.legend(loc="upper right", fontsize=14)
        plt.savefig(DIRECTORY + 'IMG_DISTANCE/frame_' + str(num) + '.jpg', dpi = 120)
        plt.close("all")
        
        # Export to a CSV
        export = [num, round(distance,2), df['Temperature'].tail(1).iloc[0]]
        #export = [round(distance,2), round(df['MA10'].tail(1).iloc[0],2), round(df['MA30'].tail(1).iloc[0],2)]
        with open(CSV_FILE, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(export)
        
        del detected_circles, circle, gs, imgOriginal, imgThreshold, imgWithFigures, imgCanny
        del x1, x2, y1, y2, x, y, r
        del csvfile
        
    print(" " + "-"*100)
    msg = "Análise finalizada, exportando gráficos..."
    print("|" + msg + " "*(100 - len(msg)) + "|")
    
    fig = plt.figure(figsize=(16,9))
    plt.plot(df['Dist'], label="Frame distance")
    plt.plot(df['MA10'], label="Moving Average 0.333 sec")
    plt.plot(df['MA30'], label="Moving Average 1 sec")
    plt.legend(loc="upper right")
    plt.savefig(DIRECTORY + 'IMG_DISTANCE/0 - Pure data.jpg', dpi = 300)
    plt.close("all")
    
    msg = "Gráficos exportados."
    print("|" + msg + " "*(100 - len(msg)) + "|")
    print(" " + "-"*100)

    del fig, minValue, maxValue, nDivision, linearSpace, msg
       