# importing the required modules
import cv2
import numpy as np

# creating final output lists
list_total_houses = []
list_total_priority = []
list_rescue_ratio = []

def task(a): #creating main function
    a = a+1
    

    for i in range(1,int(a)):

        # loading the image
        image_path = str(i) + '.png'  # Your provided image path
        image = cv2.imread(image_path)

        # converting image to HSV color space for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # defining color range for green (unburnt grass)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        # defining color range for brown (burnt grass)
        lower_brown = np.array([10, 40, 40])
        upper_brown = np.array([25, 255, 200])

        # mask the green and brown areas
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

        # dilate the masks to fill in gaps
        kernel = np.ones((5, 5), np.uint8)
        green_mask_dilated = cv2.dilate(green_mask, kernel, iterations=2)
        brown_mask_dilated = cv2.dilate(brown_mask, kernel, iterations=2)

        
        # overlay colors for regions: green for green grass, yellow for burnt grass (brown)
    
        overlay = image
        overlay[green_mask_dilated > 0] = [0, 255, 0]  # Green for unburnt grass
        overlay[brown_mask_dilated > 0] = [0, 255, 255]  # Yellow for burnt grass
            
        # this will show overlayed image 
        cv2.imshow(image_path,overlay)




        # define HSV range for red and blue houses (triangles)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # mask for red and blue houses
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # refine the red mask to only focus on the burnt region
        red_in_burnt_mask = cv2.bitwise_and(red_mask, red_mask, mask=brown_mask_dilated)

        # function to count triangles in a region using contour approximation
        def count_triangles(region_mask, house_mask):
            masked_region = cv2.bitwise_and(house_mask, house_mask, mask=region_mask)
            contours, _ = cv2.findContours(masked_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            triangle_count = 0
            for contour in contours:
                #  approximate contour to a polygon
                epsilon = 0.04 * cv2.arcLength(contour, True)  # Adjust epsilon to control contour approximation
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # filter based on contour area to avoid small or large contours
                area = cv2.contourArea(contour)
                if len(approx) == 3 and area > 50 and area < 1000:  # Adjust the area range based on your image
                    triangle_count += 1
            return triangle_count

        # count red triangles in the burnt region (yellow overlay)
        red_triangles_in_burnt = count_triangles(brown_mask_dilated, red_in_burnt_mask)

        # count red and blue houses in the green region
        red_in_green = count_triangles(green_mask_dilated, red_mask)
        blue_in_green = count_triangles(green_mask_dilated, blue_mask)

        # count red and blue houses in the burnt region
        blue_in_burnt = count_triangles(brown_mask_dilated, blue_mask)

        # calculate total priority in the burnt and green regions
        priority_burnt = red_triangles_in_burnt * 1 + blue_in_burnt * 2
        priority_green = red_in_green * 1 + blue_in_green * 2

        # calculate the rescue ratio
        if priority_green != 0:
            rescue_ratio = priority_burnt / priority_green
        else:
            rescue_ratio = float('inf')  # Avoid division by zero if priority_green is 0



        

        # Print the results
        # print(f"Number of red triangles in the burnt region: {red_triangles_in_burnt}")
        # print(f"Number of blue houses in the burnt region: {blue_in_burnt}")
        # print(f"Number of red triangles in the green region: {red_in_green}")
        # print(f"Number of blue houses in the green region: {blue_in_green}")
        # print(f"Total priority in the burnt region: {priority_burnt}")
        # print(f"Total priority in the green region: {priority_green}")
        # print(f"Rescue ratio: {rescue_ratio}")
        list_total_houses.append([red_triangles_in_burnt+blue_in_burnt, red_in_green+blue_in_green])
        list_total_priority.append([priority_burnt, priority_green])
        list_rescue_ratio.append(rescue_ratio)
        
    # printing final output lists
    print(list_total_houses)    
    print(list_total_priority)
    print(list_rescue_ratio)

    # this code is for sorting the list of images on the basis of rescue ratio
    list_sorted = list_rescue_ratio.copy()
    list_sorted.sort()
    
    for i in range(10):
        b = list_rescue_ratio[i]
        c = list_sorted.index(b)
        list_sorted[c] = str(i+1)+".png"

        
        
    list_sorted.reverse()
    print(list_sorted)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




task(10) #calling the main function and arguement is number of images you want to perform task with.