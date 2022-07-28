from cmath import inf
import cv2
from cv2 import WINDOW_FREERATIO
from cv2 import WINDOW_NORMAL
from more_itertools import random_combination
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import seaborn as sns
import pandas as pd
import skimage.feature as feature
import numpy.ma as ma
import random


#np.set_printoptions(threshold=sys.maxsize)
##SECTION 1: Functions
def manual_save():
      cv2.imwrite("/mnt/Data/Data/University of Freiburg/Hiwi_Thesis/Photos/train/input/%s.tif" % filename, resized) 
      cv2.imwrite("/mnt/Data/Data/University of Freiburg/Hiwi_Thesis/Photos/train/label/%s.tif" % filename, contour_find_copy_single_copy)
      merge = np.concatenate((resized_for_contours/255, convert_to_three_channel(contour_find_copy_single_copy)/255), axis=0) 
      cv2.imwrite("/mnt/Data/Data/University of Freiburg/Hiwi_Thesis/Photos/train/check/%s.jpeg" % filename, merge*255)


def manual_segmentation(cnt):
    plt.close('all')
    merge = np.concatenate((resized/255, convert_to_three_channel(contour_find_copy_single_copy)/255), axis=0) 
    #show_img(merge,'Before')
    fig, axs = plt.subplots(2, 1,figsize=(16,20),constrained_layout=True)
    axs[0].imshow(merge)
    axs[0].set_title('Before')
    cnt_star = np.setdiff1d(df.index.array,cnt)
    for c in cnt:
      img = images_of_contour[c].copy()
      pts_manual = points[c]
      contour_find_copy_single_copy[img!=0] = 255
    for c in cnt_star:
      img = images_of_contour[c].copy()
      pts_manual = points[c]
      contour_find_copy_single_copy[img!=0] = 0
    merge = np.concatenate((resized/255, convert_to_three_channel(contour_find_copy_single_copy)/255), axis=0) 
    axs[1].imshow(merge)
    axs[1].set_title('After')
    move_figure(fig,0,0)  
    #show_img(merge,'After')

def show_img (arr,*title):
  fig1 = plt.figure(figsize=(8,10))
  plt.imshow(arr)
  for t in title:
    title = t
  plt.title(title)
  plt.show(block=False)
  move_figure(fig1,0,0)



def _get_slice_bbox(arr):
    nonzero = np.nonzero(arr)
    return [(min(a), max(a)+1) for a in nonzero]

def crop(arr):
    slice_bbox = _get_slice_bbox(arr)
    return arr[tuple([slice(*a) for a in slice_bbox])]

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def detect_horizontal_lines(image,separator_columns):
    array_zero = []
    (_, image) = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV)
    image = 255 - image
    separator_columns = np.array(separator_columns) - separator_columns[0]
    for col in separator_columns:
        if col == separator_columns[-1]:
            image[:,col - 1] = np.zeros_like(image[:,col-1])
            break
        image[:,col] = np.zeros_like(image[:,col])
    #! Below Block only detects WHOLE horizontal lines. However it does not work if even one pixel is non zero
    # for row in range(image.shape[0]):
    #     a = image[row,:]
    #     if all(not v for v in a):
    #         array_zero.append(row)
    #!Below Block is an improved version of above algorithm. It works if 90% of the horizontal line is black
    for row in range(image.shape[0]):
        a = image[row,:]
        if np.count_nonzero(a==0)/a.shape[0] >= 0.9 and row > 225 and row < 525:
            array_zero.append(row)

    separator_row = []
    for i in range(len(array_zero)):
        
        if array_zero[i] > 300 and array_zero[i+1] > array_zero[i] + 50:
            separator_row.append(array_zero[i])
            separator_row.append(array_zero[i+1])
            break
    array_zero = separator_row
    return array_zero

def detect_vertical_lines(image):
    array_zero = []
    for col in range(image.shape[1]):
        a = image[:,col]
        if all(not v for v in a):
            array_zero.append(col)

    return array_zero

def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)

def delete_small_contour(contours,threshold_area):
    small_contours_index = np.where(np.array([cv2.contourArea(contours[i]) for i in range(len(contours))]) < threshold_area)[0]
    if len(small_contours_index) > 0:
        contours = np.delete(contours,small_contours_index)
    return contours

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)

def convert_to_three_channel(image):
  dim_3_image = np.zeros((image.shape[0],image.shape[1],3))
  dim_3_image[:,:,0] = image
  dim_3_image[:,:,1] = image
  dim_3_image[:,:,2] = image

  return dim_3_image

def convert_to_one_channel(image,i):
  single_channel = image[:,:,i]

  return single_channel
##SECTION 2: Reading Background Images and Videos and Rotation of Background Images
back1 = cv2.imread('images/snapshot1.bmp',0)
back2 = cv2.imread('images/snapshot2.bmp',0)
back3 = cv2.imread('images/snapshot3.bmp',0)
back4 = cv2.imread('images/snapshot4.bmp',0)
back1 = rotate_image(back1.copy(),0.50132528)
back2 = rotate_image(back2.copy(),0.50132528)
back3 = rotate_image(back3.copy(),0.50132528)
back4 = rotate_image(back4.copy(),0.50132528)
# Create a video capture object, in this case we are reading the video from a file

#% ENTER VIDEO AND BACKGROUND NAME DETAILS BELOW
vid_capture = cv2.VideoCapture('Videos/4.mp4')
frame_background = back4
#% ENTER VIDEO AND BACKGROUND NAME DETAILS BELOW
success = False
flag_optimize = False
diff = 1000
frameSize = (791,113)

out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, frameSize)


##SECTION 3: Resizing of Background Image to Focus on only two sections
##! Resize Background Image
    #*Background Image
intermediate_crop_back = frame_background[325:460].copy()
(_, intermediate_bw_back) = cv2.threshold(intermediate_crop_back, 190, 255, cv2.THRESH_BINARY_INV)
zero_lines_column_background = detect_vertical_lines(intermediate_bw_back)
degree_background_rotation = 0.1
while len(zero_lines_column_background) < 3:
    intermediate_bw_back = rotate_image(intermediate_bw_back,0.1)
    (_, intermediate_bw_back) = cv2.threshold(intermediate_bw_back, 250, 255, cv2.THRESH_BINARY_INV)
    intermediate_bw_back = 255 - intermediate_bw_back
    degree_background_rotation += 1
    zero_lines_column_background = detect_vertical_lines(intermediate_bw_back)
    if len(zero_lines_column_background)>= 3:
        break
##NOTE: Below If Statement is Done to Filter Out the Inlet
if np.array_equal(back1,frame_background):
    zero_lines_column_background[-1] = zero_lines_column_background[-1] - 100
    zero_lines_column_background = np.array(zero_lines_column_background)
    if zero_lines_column_background[-1] <= zero_lines_column_background[-2]:
      zero_lines_column_background = np.delete(zero_lines_column_background,(zero_lines_column_background > zero_lines_column_background[-1]).nonzero()[0][0])
#!splice_vertical_image = frame_background[300:500,zero_lines_column_background[0]:zero_lines_column_background[-1]]
splice_vertical_image = frame_background[:,zero_lines_column_background[0]:zero_lines_column_background[-1]]
zero_lines_row_background = detect_horizontal_lines(splice_vertical_image,zero_lines_column_background)
final_splice_background = splice_vertical_image[zero_lines_row_background[0]:zero_lines_row_background[1],:]
##! Resize Background Image End


if (vid_capture.isOpened() == False):
  print("Error opening the video file")
# Read fps and frame count
else:
  # Get frame rate information
  # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
  fps = vid_capture.get(5)
  print('Frames per second : ', fps,'FPS')
  # Get frame count
  # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
  frame_count = vid_capture.get(7)
  print('Frame count : ', frame_count)
while(vid_capture.isOpened()):
  # vid_capture.read() methods returns a tuple, first element is a bool
  # and the second is frame
  ret, frame = vid_capture.read()
  scale_percent = 50 # percent of original size
  
  ##SECTION 4 : Video Reading begins here
  if ret == True:
    ##NOTE: Rotation of Video Frame Below
    frame = rotate_image(frame.copy(),0.50132528)
    array_zero_rows = []
    
    while diff > 5 or flag_optimize == False:
      for i in range(int(vid_capture.get(7)) - 1):
        vid_capture.set(cv2.CAP_PROP_POS_FRAMES, i+1)
        status , ith_frame = vid_capture.read()
        ith_frame = rotate_image(ith_frame.copy(),0.50132528)
        ##SECTION 5 : Resizing of Video Frame to focus on two sections bounded insided separator
        ##! Resize Block : This block focuses on just extracting the relevant part of the frame in the video
        intermediate_crop_video = ith_frame[325:460].copy()
        ##NOTE: I changed threshold from 190 to 170 in video (17/05/2022). It is still 190 in background resize
        (_, intermediate_bw_video) = cv2.threshold(intermediate_crop_video, 170, 255, cv2.THRESH_BINARY_INV)
        #*Video Frame
        intermediate_bw_video = convert_to_one_channel(intermediate_bw_video,0)
        if success == False:
          zero_lines_column_video = detect_vertical_lines(intermediate_bw_video)
        degree_video_rotation = 0
        ##SECTION 6: Finding the Optimum Frame in Video as compared to Background Image. We then use the Optimum Frame to find the Horizontal Lines
        while len(zero_lines_column_video)  < 3 and success == False:
            intermediate_bw_video = rotate_image(intermediate_bw_video,0.1)
            intermediate_crop_video = rotate_image(intermediate_crop_video,0.1)
            (_, intermediate_bw_video) = cv2.threshold(intermediate_bw_video, 250, 255, cv2.THRESH_BINARY_INV)
            intermediate_bw_video = 255 - intermediate_bw_video
            degree_video_rotation += 0.1
            zero_lines_column_video = detect_vertical_lines(intermediate_bw_video)
            if len(zero_lines_column_video)>= 3:
                success = True
                break
        ##NOTE: Below If Statement is Done to Filter Out the Inlet
        if np.array_equal(back1,frame_background):
            zero_lines_column_video[-1] = zero_lines_column_video[-1] - 100
            zero_lines_column_video = np.array(zero_lines_column_video)
            if zero_lines_column_video[-1] <= zero_lines_column_video[-2]:
              zero_lines_column_video = np.delete(zero_lines_column_video,(zero_lines_column_video > zero_lines_column_video[-1]).nonzero()[0][0])
        #!splice_vertical_video = frame[300:500,zero_lines_column_video[0]:zero_lines_column_video[-1]]
        splice_vertical_video = ith_frame[:,zero_lines_column_video[0]:zero_lines_column_video[-1]]
        splice_vertical_video = convert_to_one_channel(splice_vertical_video,0)
        zero_lines_row_video = detect_horizontal_lines(splice_vertical_video,zero_lines_column_video)
        array_zero_rows.append(zero_lines_row_video)
        print("Frame "+ str(i+1)+ " has the value " + str(zero_lines_row_video))
        diff = np.diff(zero_lines_row_background)[0] - np.diff(zero_lines_row_video)[0]
        threshold_value = i
        if diff <= 1:
          print("Value under the Threshold found as "+ str(zero_lines_row_video))
          vid_capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
          break
        #final_splice_video = splice_vertical_video[zero_lines_row_video[0]:zero_lines_row_video[1],:]
      flag_optimize = True
      zero_lines_row_video = array_zero_rows[np.argmax([[np.diff(array_zero_rows[i]) for i in range(len(array_zero_rows))][i][0] for i in range(len(array_zero_rows))])]
        ##!Resize Block End
    
    final_splice_background = frame_background[zero_lines_row_background[0]:zero_lines_row_background[1],zero_lines_column_background[0]:zero_lines_column_background[-1]]
    final_splice_video = frame[zero_lines_row_video[0]:zero_lines_row_video[1],zero_lines_column_video[0]:zero_lines_column_video[-1]]
    ##NOTE: We need to convert 3 channel frame image to single channel frame image
    #final_splice_video = final_splice_video[:,:,0]
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    resized_background = cv2.resize(frame_background, dim, interpolation = cv2.INTER_AREA)

    ##!This was done during Norway Trip. This block ensures equality of shapes of video and background
    ##NOTE: Done during Norway trip
    (_, final_bw_back) = cv2.threshold(final_splice_background, 190, 255, cv2.THRESH_BINARY_INV)
    zero_lines_column_final_background = np.array(detect_vertical_lines(final_bw_back))
    (_, final_bw_video) = cv2.threshold(convert_to_one_channel(final_splice_video,0), 190, 255, cv2.THRESH_BINARY_INV)
    zero_lines_column_final_video = np.array(detect_vertical_lines(final_bw_video))
    zero_lines_column_final_background = zero_lines_column_final_background[zero_lines_column_final_background!=0]
    zero_lines_column_final_video = zero_lines_column_final_video[zero_lines_column_final_video!=0]
    for video in zero_lines_column_final_video:
        for back in zero_lines_column_final_background:
            if video - back == final_splice_video.shape[1] - final_splice_background.shape[1]:
              final_splice_video = final_splice_video[:,(video - back):]
    if final_splice_background.shape[0] - final_splice_video.shape[0] != 0 :
      delta = final_splice_background.shape[0] - final_splice_video.shape[0]
      final_splice_background = final_splice_background[delta:,:]
    ##!This block ensures equality of shapes of video and background


    ##! NOW THAT THE SHAPES OF VIDEO AND BACKGROUND ARE SAME
    resized = final_splice_video.copy()
    resized_for_contours = resized.copy()

    resized_copy = resized.copy()
    ##!TEST 2nd June 2022
    #xx = hist_match(resized,convert_to_three_channel(resized_background))


    ##!TEST 2nd June 2022
    resized_background = final_splice_background
    ##! NOW THAT THE SHAPES OF VIDEO AND BACKGROUND ARE SAME
    resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    img = resized_gray



    # threshold_image = gammaCorrection(convert_to_one_channel(resized,0), 5.1)
    # threshold_image[threshold_image<223] = 0
    # threshold_image = gammaCorrection(threshold_image, 4.1)
    # threshold_image = gammaCorrection(threshold_image, 0.1)
    # threshold_image = cv2.GaussianBlur(threshold_image,(11,11),0)
    # threshold_image[threshold_image<189] = 0
    # threshold_image[threshold_image!=0] = 255


    #!NEW
    threshold_image = gammaCorrection(convert_to_one_channel(resized,0) , 1)
    threshold_image[threshold_image<130] = 0
    threshold_image = gammaCorrection(threshold_image, 4.1)
    #threshold_image = gammaCorrection(threshold_image, 0.1)
    threshold_image_inter = cv2.GaussianBlur(threshold_image,(5,1),0)
    threshold_image[threshold_image_inter<210] = 0 
    #threshold_image[threshold_image<189] = 0
    threshold_image[threshold_image!=0] = 255
    #!NEW
    contour_find = threshold_image
    contours,_ = cv2.findContours(contour_find, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea)
    contour_find_copy_single = contour_find.copy()
    
    contour_find_RGB = np.zeros_like(resized)
    contour_find_RGB[:,:,0] = contour_find_copy_single
    contour_find_RGB[:,:,1] = contour_find_copy_single
    contour_find_RGB[:,:,2] = contour_find_copy_single



    contour_find_RGB_copy = contour_find_RGB.copy()   
    lst_intensities = []
    background_lst_intensities = []
    images_of_contour = []
    background_images_of_contour = []
    points = []
    mask_grey_list = []
        # For each list of contour points...
    for i in range(len(contours)):
        # Create a mask image that contains the contour filled in
        mask_image = np.zeros_like(contour_find_RGB_copy)
        cv2.drawContours(mask_image, contours, i, color=(255,255,255), thickness=-1)
        # Access the image pixels and create a 1D numpy array then add to list
        mask_image_grey = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        pts = np.where(mask_image_grey == 255)
        points.append(pts)
        #Below line gives RGB intensities. Let me just take the mean over axis=1
        pixel_values = np.mean(resized[pts[0], pts[1]],axis=1)
        lst_intensities.append(pixel_values)
        background_pixel_values = resized_background[pts[0], pts[1]]
        background_lst_intensities.append(background_pixel_values)
        intermediate_mask = mask_image_grey
        intermediate_mask[intermediate_mask==255] = 1
        images_of_contour.append(intermediate_mask * resized_gray)
        background_images_of_contour.append(intermediate_mask * resized_background)
        mask_grey_list.append(mask_image_grey)

    
    ##SECTION 7:Debug Commands 
    ##* No.1
    #  for i in range(len(contours)):
        # fig, (ax1, ax2) = plt.subplots(1,2)
        # plt.title("Contour" + str(i+1))
        # ax1.imshow(images_of_contour[i])
        # ax2.hist(lst_intensities[i])
    
    if vid_capture.get(1) == 663:
      dummy_variable = 1
    ##*No.2
    # histogram_list = []
    # for i in range(len(contours)):
    #     img  =  images_of_contour[i]
    #     hist = cv2.calcHist([img], [0], mask_grey_list[i], [256], [0,256])
    #     fig, (ax1, ax2) = plt.subplots(1,2,figsize =(4,2))
    #     move_figure(fig,100,400)
    #     plt.title("Contour" + str(i+1))
    #     ax1.axis("off")
    #     ax1.imshow(img)
    #     ax2.set_xlim([0, 256])
    #     ax2.plot(hist)
    #     histogram_list.append(hist)
    # img  =  resized_background
    # hist = cv2.calcHist([img], [0], mask_grey_list[i], [256], [0,256])
    # fig, (ax1, ax2) = plt.subplots(1,2,figsize=(4,2))
    # move_figure(fig,100,400)
    # plt.title("Background")
    # ax1.axis("off")
    # ax1.imshow(img)
    # ax2.set_xlim([0, 256])
    # ax2.plot(hist)
    # # histogram_list.append(hist)
    # fig, (ax1) = plt.subplots(1,1,figsize=(4,2))
    # move_figure(fig,100,400)
    # #ax1.title("Original Frame")
    # ax1.imshow(resized)
    # p = [cv2.compareHist(histogram_list[-1],histogram_list[i], cv2.HISTCMP_INTERSECT) for i in range(len(histogram_list))]
    
    #Making Histograms For each 1st Stage Contour
    histogram_list = []
    background_histogram_list=[]
    for i in range(len(contours)):

        hist_video = cv2.calcHist([images_of_contour[i]], [0], mask_grey_list[i], [256], [0,256])
        histogram_list.append(hist_video)
        hist_background = cv2.calcHist([background_images_of_contour[i]], [0], mask_grey_list[i], [256], [0,256])
        background_histogram_list.append(hist_background)

        ##!Only Use in Debugging
        # correlation_values = [(cv2.compareHist(background_histogram_list[i],histogram_list[i], cv2.HISTCMP_CORREL)*100) for i in range(len(histogram_list))]
        # correlation_values_KL = [(cv2.compareHist(background_histogram_list[i],histogram_list[i], cv2.HISTCMP_KL_DIV))/np.sum(histogram_list[i]) for i in range(len(histogram_list))]
        # max_KL = 1
        # correlation_values_KL = np.array(correlation_values_KL)/1
        # if correlation_values[i] > 0:
        #   fig, axs = plt.subplots(2,2,figsize=(6,2))
        #   move_figure(fig,100,-1000)
        #   axs[0,0].imshow(background_images_of_contour[i])
        #   axs[0,0].set_title("Background")
        #   axs[1,0].imshow(images_of_contour[i])
        #   axs[1,0].set_title("Video")
        #   axs[0,1].plot(background_histogram_list[i])
        #   axs[1,1].plot(histogram_list[i])
        #   fig.suptitle(" Contour " + str(i+1) + " Closeness = " + 
        #   str("{:.2f}".format(cv2.compareHist(background_histogram_list[i],histogram_list[i], cv2.HISTCMP_CORREL) * 100)) + "% and norm KL" + str("{:.2f}".format(correlation_values_KL[i])) )        
          ##!Only Use in Debugging End

        ##! Use this before above block
        # correlation_values = [(cv2.compareHist(background_histogram_list[i],histogram_list[i], cv2.HISTCMP_CORREL)*100) for i in range(len(histogram_list))]
        # correlation_values_KL = [(cv2.compareHist(background_histogram_list[i],histogram_list[i], cv2.HISTCMP_KL_DIV))/np.sum(histogram_list[i]) for i in range(len(histogram_list))]
        # max_KL = 1
        # correlation_values_KL = np.array(correlation_values_KL)/1
        # correlation_values = np.array(correlation_values)
        # sns.set_style("darkgrid")
        # plt.scatter(correlation_values_KL,correlation_values)
        # plt.show()
        ##!Use this before above block

    correlation_values = [(cv2.compareHist(background_histogram_list[i],histogram_list[i], cv2.HISTCMP_CORREL)*100) for i in range(len(histogram_list))]
    area_1st_stage = [cv2.contourArea(contours[i]) for i in range(len(contours))]
    correlation_values_KL = [(cv2.compareHist(background_histogram_list[i],histogram_list[i], cv2.HISTCMP_KL_DIV))/np.sum(histogram_list[i]) for i in range(len(histogram_list))]
    correlation_values_KL = np.array(correlation_values_KL)
    correlation_values = np.array(correlation_values)
    df = pd.DataFrame({'Correl': correlation_values,
                        'Area': area_1st_stage,
                        'KL Area Norm': correlation_values_KL,
                        #'CHISQR Area Normalized': [(cv2.compareHist(background_histogram_list[i],histogram_list[i], cv2.HISTCMP_CHISQR))/np.sum(histogram_list[i]) for i in range(len(histogram_list))],
                        #'Bhattacharya': [(cv2.compareHist(background_histogram_list[i],histogram_list[i], cv2.HISTCMP_BHATTACHARYYA)) for i in range(len(histogram_list))],
                        #'CHISQRALT Area Normalized': [(cv2.compareHist(background_histogram_list[i],histogram_list[i], cv2.HISTCMP_CHISQR_ALT))/np.sum(histogram_list[i]) for i in range(len(histogram_list))],
                        #'Intersect': [(cv2.compareHist(background_histogram_list[i],histogram_list[i], cv2.HISTCMP_INTERSECT)) for i in range(len(histogram_list))],
                        #'Hellinger': [(cv2.compareHist(background_histogram_list[i],histogram_list[i], cv2.HISTCMP_HELLINGER)) for i in range(len(histogram_list))],
                        #! I am just going to comment the part with masking arrays since its taking too much computation
                        #'Var1': [ma.masked_array(crop(images_of_contour[i]), mask=1 - crop(mask_grey_list[i])).var() for i in range(len(contours))],
                        #'Back Var1': [ma.masked_array(crop(background_images_of_contour[i]), mask=1 - crop(mask_grey_list[i])).var() for i in range(len(contours))],
                        #'Mean1': [ma.masked_array(crop(images_of_contour[i]), mask=1 - crop(mask_grey_list[i])).mean() for i in range(len(contours))],
                        #'Back Mean1': [ma.masked_array(crop(background_images_of_contour[i]), mask=1 - crop(mask_grey_list[i])).mean() for i in range(len(contours))],
                        'Var': [np.var(lst_intensities[i]) for i in range(len(contours))],
                        'Back Var': [np.var(background_lst_intensities[i]) for i in range(len(contours))],
                        'Mean': [np.mean(lst_intensities[i]) for i in range(len(contours))],
                        'Back Mean': [np.mean(background_lst_intensities[i]) for i in range(len(contours))],
                        'Median': [np.median(lst_intensities[i]) for i in range(len(contours))],
                        'Back Median': [np.median(background_lst_intensities[i]) for i in range(len(contours))],
                        'Mean X Pos': [points[i][1].mean() for i in range(len(contours))],
                        'Mean Y Pos': [points[i][0].mean() for i in range(len(contours))],
                        },)
    df = df.round(decimals=1)

    remove_contour = np.where(np.array(correlation_values) < 85)[0]
    contour_find_copy_single_copy = contour_find_copy_single.copy()
    ## Removing Contours which have correlation values lesser than 85%
    for i in range(len(remove_contour)):
      # #! NEW 18.07.2022
      # if cv2.contourArea(contours[remove_contour[i]]) <= 10:
      #   contour_find_copy_single_copy[images_of_contour[remove_contour[i]]!=0]=255
      # else:
      contour_find_copy_single_copy[images_of_contour[remove_contour[i]]!=0]=0
      # #! NEW 18.07.2022
    ## Let's draw new contours
    contours_2nd_stage,hierarchy_2nd_stage = cv2.findContours(contour_find_copy_single_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_2nd_stage = sorted(contours_2nd_stage, key=cv2.contourArea)

    #!GLCM
    # color = ['bo','go','ro','co','mo','yo','ko','wo']
    # fig = plt.figure(figsize=(8, 8))
    # axs = fig.subplots(4)
    # ax = axs[0]
    # bx = axs[1]
    # cx = axs[2]
    # dx = axs[3]
    # c=0
    # for i in df[df['Area']>1000].index:
    #   PATCH_SIZE = 35

    #   image = resized[:,:,0]
    #   #plt.imshow(image, cmap='gray')

    #   #Full image
    #   GLCM = feature.graycomatrix(image, [10], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    #   a= feature.graycoprops(GLCM, 'energy')[0, 0]

    #   # select some patches from grassy areas of the imagea
    #   #bubble_locations = [(59,300),(21,471), (51,559), (46,711), (81,744)]
    #   indices = np.where(images_of_contour[i]>  200)
    #   rand = random.sample((range(1, len(indices[0]))), 5)
    #   coordinates = [(indices[0][i],indices[1][i]) for i in rand]
    #   bubble_patches = []
    #   background_patches = []
    #   for loc in coordinates:
    #       bubble_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
    #                                 loc[1]:loc[1] + PATCH_SIZE])
    #       background_patches.append(resized_background[loc[0]:loc[0] + PATCH_SIZE,
    #                                 loc[1]:loc[1] + PATCH_SIZE])
    #   # compute some GLCM properties each patch
    #   diss_sim = []
    #   corr = []
    #   homogen = []
    #   energy = []
    #   contrast = []
    #   for patch in (bubble_patches):
    #       glcm = feature.graycomatrix(patch, distances=[10], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256,
    #                           symmetric=True, normed=True)
    #       glcm[0][0] = np.zeros_like(glcm[0][0])
    #       diss_sim.append(feature.graycoprops(glcm, 'dissimilarity')[0, 0]) #[0,0] to convert array to value
    #       corr.append(feature.graycoprops(glcm, 'correlation')[0, 0])
    #       homogen.append(feature.graycoprops(glcm, 'homogeneity')[0, 0])
    #       energy.append(feature.graycoprops(glcm, 'energy')[0, 0])
    #       contrast.append(feature.graycoprops(glcm, 'contrast')[0, 0])
    #   diss_sim_back = []
    #   corr_back = []
    #   homogen_back = []
    #   energy_back = []
    #   contrast_back = []
    #   for patch in (background_patches):
    #       glcm = feature.graycomatrix(patch, distances=[10], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256,
    #                           symmetric=True, normed=True)
    #       glcm[0][0] = np.zeros_like(glcm[0][0])
    #       diss_sim_back.append(feature.graycoprops(glcm, 'dissimilarity')[0, 0]) #[0,0] to convert array to value
    #       corr_back.append(feature.graycoprops(glcm, 'correlation')[0, 0])
    #       homogen_back.append(feature.graycoprops(glcm, 'homogeneity')[0, 0])
    #       energy_back.append(feature.graycoprops(glcm, 'energy')[0, 0])
    #       contrast_back.append(feature.graycoprops(glcm, 'contrast')[0, 0])
      
    #   # display original image with locations of patches
      
    #   ax.imshow(image, cmap=plt.cm.gray,
    #             vmin=0, vmax=255)
      
    #   for (y, x) in coordinates:
    #       ax.plot(x, y,color[c])
    
    #   cx.imshow(resized_background, cmap=plt.cm.gray,
    #             vmin=0, vmax=255)
      
    #   for (y, x) in coordinates:
    #       cx.plot(x, y,color[c])
    #   #for (y, x) in fluid_locations:
    #       #ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
      
    #   ax.set_xlabel('Original Image')
    #   ax.set_xticks([])
    #   ax.set_yticks([])
    #   ax.axis('image')
      
    #   bx.plot(energy, corr, color[c],
    #           label='Contour'+str(i)+ ' Var Con ' + str(np.var(energy)))
        
    #   dx.plot(energy_back, corr_back, color[c],
    #           label='Contour'+str(i)+ ' Var Con ' + str(np.var(energy_back)))
      
    #   c = c+1
    #   #ax.plot(diss_sim, corr, 'bo',
    #   #        label='Fluid')
    #   bx.set_xlabel('GLCM Dissimilarity')
    #   bx.set_ylabel('GLCM Correlation')
    #   bx.legend()
    #   dx.legend()

    #!
    
    #Let's Remove very small contours
    threshold_area = 40
    small_contours_index = np.where(np.array([cv2.contourArea(contours_2nd_stage[i]) for i in range(len(contours_2nd_stage))]) < threshold_area)[0]
    if len(small_contours_index) > 0:
      for i in range(len(small_contours_index)):
        mask_image_small = np.zeros_like(contour_find_RGB_copy)
        cv2.drawContours(mask_image_small, contours_2nd_stage, small_contours_index[0], color=(1,1,1), thickness=-1)
        mask_image_small = mask_image_small * np.ones_like(resized)*255
        contour_find_copy_single_copy = convert_to_one_channel(mask_image_small,0) + contour_find_copy_single_copy
      if len(contours_2nd_stage) == 1:
        contours_2nd_stage = np.delete(contours_2nd_stage,[0,1])
      else:
        contours_2nd_stage = np.delete(contours_2nd_stage,small_contours_index)
    #Draw Contours on Frames

    cv2.drawContours(resized_for_contours, contours_2nd_stage, -1, (255,0,0), 2)
    cv2.drawContours(contour_find_RGB, contours, -1, (0,0,255), 1)
    area = [cv2.contourArea(contours_2nd_stage[i]) for i in range(len(contours_2nd_stage))]
    if 'area_value' in locals():
      dummy_variable = 1 #dummy line
    else:
      area_value = []
    area_value.append(np.sum(area)/(resized.shape[0] * resized.shape[1])*100)
    if len(area_value) == 21:
      area_value = area_value[1:]
    total_area = 'Void Ratio = ' + str("{:.2f}".format(np.median(area_value)))
    
    

    # #!Below Part saves Random Frames
    random_frame = np.random.randint(0,int(vid_capture.get(7)),int(vid_capture.get(7)*0.7))
    back_list = [back1, back2, back3, back4]
    back_list = [np.array_equal(i,frame_background) for i in back_list]
    index = np.where(back_list)[0][0] + 1
    filename = str(index) + '_' + str(int(vid_capture.get(1)))
  
    # # if int(vid_capture.get(1)) in random_frame and os.path.exists("/mnt/Data/Data/University of Freiburg/Hiwi_Thesis/Photos/train/check_done/%s.jpeg" % filename) == False :
    # if os.path.exists("/mnt/Data/Data/University of Freiburg/Hiwi_Thesis/Photos/train/input_done4/%s.tif" % filename) == True :
    #   cv2.imwrite("/mnt/Data/Data/University of Freiburg/Hiwi_Thesis/Photos/train/input/%s.tif" % filename, resized) 
    #   cv2.imwrite("/mnt/Data/Data/University of Freiburg/Hiwi_Thesis/Photos/train/label/%s.tif" % filename, contour_find_copy_single_copy)
    #   merge = np.concatenate((resized_for_contours/255, convert_to_three_channel(contour_find_copy_single_copy)/255), axis=0) 
    #   cv2.imwrite("/mnt/Data/Data/University of Freiburg/Hiwi_Thesis/Photos/train/check/%s.jpeg" % filename, merge*255)
    #   cv2.imwrite("/mnt/Data/Data/University of Freiburg/Hiwi_Thesis/Photos/train/frame/%s.jpeg" % filename, frame)

    ##! Below Part is used for manual segmentation
    # if vid_capture.get(1) >= 1:
    #   dummy_variable = 1
    #   if os.path.exists("/mnt/Data/Data/University of Freiburg/Hiwi_Thesis/Photos/train/input_done4/%s.tif" % filename) == False:
    #     plt.figure(figsize=(20,10))
    #     plt.close('all')
    #     merge = np.concatenate((resized_for_contours/255, convert_to_three_channel(contour_find_copy_single_copy)/255), axis=0) 
    #     show_img(merge, 'Existing Final Output'+ str(filename))
    #     print(df[df['Area'] > 40])
    #     # j = len(df[df['Area'] > 10])
    #     # for i in df[df['Area'] > 10].index:
    #     #   show_img(images_of_contour[i],i)
    #     fig, axs = plt.subplots(len(df[df['Area'] > 40]), 1, figsize = (10,20),constrained_layout=True)
    #     for e, i in enumerate(df[df['Area'] > 40].index):
    #         axs[e].imshow(images_of_contour[i])
    #         axs[e].set_title(i)
    #     plt.show(block=False)
    #     move_figure(fig,850,0)
    #     dummy_variable = 1
    
    ##!Below Line puts text on 'Original'
    cv2.putText(resized_for_contours, str(total_area), (15, 15),
    cv2.FONT_HERSHEY_TRIPLEX, 0.8 , (0,0,255),2)
    Window_Name = 'Final Output'
    cv2.namedWindow(Window_Name,WINDOW_NORMAL)
    cv2.resizeWindow(Window_Name,900,200)
    cv2.moveWindow(Window_Name,100,810)
    cv2.namedWindow('Original',WINDOW_NORMAL)
    cv2.resizeWindow('Original',900,200)
    cv2.moveWindow('Original',100,-1000)
    cv2.imshow('Final Output',contour_find_copy_single_copy)
    cv2.imshow('Original',resized_for_contours)
    cv2.namedWindow('Intermediate Stage',WINDOW_NORMAL)
    cv2.resizeWindow('Intermediate Stage',900,200)
    cv2.moveWindow('Intermediate Stage',100,570)
    cv2.imshow('Intermediate Stage',contour_find_RGB)
    #cv2.imshow('Intermediate Stage',cv2.normalize(magnitude_spectrum,None,0,1,cv2.NORM_MINMAX))
    cv2.namedWindow('Frame',WINDOW_NORMAL)
    cv2.moveWindow('Frame',1100,2450)
    cv2.resizeWindow('Frame',780,500)
    cv2.imshow('Frame',frame)
    cv2.namedWindow('Background',WINDOW_NORMAL)
    cv2.moveWindow('Background',1100,-1000)
    cv2.resizeWindow('Background',780,500)
    cv2.imshow('Background',resized_background)
    
    #cv2.namedWindow('Background',WINDOW_NORMAL)
    #cv2.moveWindow('Background',100,500)
    #cv2.imshow('Background',frame_background)
    
    aa = vid_capture.get(1)
    
    # 20 is in milliseconds, try to increase the value, say 50 and observe
    key = cv2.waitKey(20)
    if key == ord('q'):
      break
  else:
    break
# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()


#CANNY
  # src = resized
  # src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  # low_threshold = 10
  # ratio=3
  # kernel_size =3
  # #img_blur = cv.blur(src_gray, (3,3))
  # detected_edges = cv2.Canny(src_gray, low_threshold, low_threshold*ratio, kernel_size)
  # mask = detected_edges != 0
  # dst = src * (mask[:,:,None].astype(src.dtype))
  # plt.imshow(dst)
  # plt.show()


  #! Check the segment was automatically segmented
  # os.path.exists("/mnt/Data/Data/University of Freiburg/Hiwi_Thesis/Photos/train/input_done1/%s.tif" % filename)
  #! Show merged image file
  #merge = np.concatenate((resized_for_contours/255, convert_to_three_channel(contour_find_copy_single_copy)/255), axis=0)
  #show_img(merge)

