def undistort(image):
    return cv2.undistort(image, mtx, dist, None, mtx)

def thresh(img, thresh=(0,255)):
    binary = np.zeros_like(img)
    binary[(img > thresh[0]) & (img <= thresh[1])] = 1
    return binary

def colorthresh(img):
    r_channel = img[:,:,2]
    r_thresh = thresh(r_channel, thresh=(200,255))
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_thresh = thresh(s_channel, thresh=(90,255))
    color_combined = np.zeros_like(s_thresh)
    color_combined[((r_thresh==1)|(s_thresh==1))] = 1
    return color_combined

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sbinary

def mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 200)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=9, thresh=(0.7, 1.3)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def gradthresh(img):
    gradx = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)
    grady = abs_sobel_thresh(img, orient='y', thresh_min=20, thresh_max=100)
    mag_binary = mag_thresh(img)
    dir_binary = dir_threshold(img)

    grad_combined = np.zeros_like(dir_binary)
    grad_combined[((gradx==1)|(mag_binary == 1)) & ((grady == 1)|(mag_binary == 1))] = 1

    return grad_combined

def combinethresh(grad_thresh, color_thresh):
    combined_color_grad = np.zeros_like(grad_thresh)
    combined_color_grad[(grad_thresh==1) & (color_thresh==1)] = 1
    return combined_color_grad

def warp(img):
    img_size = (img.shape[1], img.shape[0])

    #top right, bottom right, bottom left, top left
    src = np.float32([[490, 482],[810, 482], [1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0], [1250, 720],[40, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped_img, M, Minv

def find_lane(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9
    window_height = np.int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return ploty, lefty, righty, leftx, rightx, left_fitx, right_fitx

def calculate_curvature_offset(ploty, lefty, righty, leftx, rightx, w, h):
    y_eval = np.max(ploty)
    lane_width_pix = w * 0.57
    ym_per_pix = 23/h # meters per pixel in y dimension
    xm_per_pix = 3.7/lane_width_pix # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    curvrad = np.mean([left_curverad, right_curverad])

    image_centre = lane_width_pix * xm_per_pix / 2

    vehicle_centre = np.mean([left_fit_cr, right_fit_cr])

    vehicle_offset = abs(image_centre - vehicle_centre)

    return curvrad, vehicle_offset

def draw_lane(binary_warped, undist, Minv, ploty, left_fitx, right_fitx, curverad, vehicle_offset):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)

    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    img_size = (color_warp.shape[1], color_warp.shape[0])

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size)

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


    cv2.putText(result, 'Distance from centre: {:.2f} m'.format(vehicle_offset), (100,80), fontFace = 16,\
                    fontScale = 2, color=(255,255,255), thickness = 4)

    cv2.putText(result, 'Radius of Curvature {} m'.format(int(curverad)), (120,140),
             fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 4)

    return result

def pipeline(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undist = undistort(img)
    print(undist.shape[1])
    color_thresholded = colorthresh(undist)
    grad_thresholded = gradthresh(color_thresholded)
    combined_thresholded = combinethresh(grad_thresholded, color_thresholded)
    warped_img, M, Minv = warp(combined_thresholded)
    ploty, lefty, righty, leftx, rightx, left_fitx, right_fitx = find_lane(warped_img)
    curverad, vehicle_offset = calculate_curvature_offset(ploty, lefty, righty, leftx, rightx, undist.shape[1], undist.shape[0])
    result = draw_lane(warped_img, undist, Minv, ploty, left_fitx, right_fitx, curverad, vehicle_offset)
    return result
