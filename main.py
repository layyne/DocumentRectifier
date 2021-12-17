import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from skimage.io import imread
import matplotlib.pyplot as plt


# Given a numpy array of 2D quadrangle corners,
# rearranges the coordinates into the order
#   [top-left, top-right, bottom-right, bottom-left]
# so that functions dependent on the orientation of the
# polygon formed by the corners will work properly.
def fixOrientation(pts):
    # Only consider one set of corners
    pts_trunc = pts[:4]

    # Compute centroid
    center = np.mean(pts_trunc, axis=0)

    tops = []
    bottoms = []

    # Split into top/bottom corners
    for pt in pts_trunc:
        if pt[1] < center[1]:
            tops.append(pt)
        else:
            bottoms.append(pt)

    # Split top corners left and right
    tl, tr = (tops[0], tops[1]) if tops[0][0] < tops[1][0] else (
        tops[1], tops[0])

    # Split bottom corners left and right
    bl, br = (bottoms[0], bottoms[1]) if bottoms[0][0] < bottoms[1][0] else (
        bottoms[1], bottoms[0])

    return np.array([tl, tr, br, bl])


# Given an image containing a document, (hopefully) returns a 2D numpy array
# containing the coordinates of the corners of the document
def detectCorners(img):
    # Shrink image for faster computation
    # Also Gaussian kernel size 11x11 doesn't work on really large images
    resize = cv2.resize(img.copy(), (img.shape[1]//5, img.shape[0]//5))
    # Convert to grayscale
    gray = cv2.cvtColor(resize, cv2.COLOR_RGB2GRAY)
    # Smooth image
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Canny edge detection
    # Lower threshold was determined empirically
    # (i.e. I messed around with the value until it worked)
    can = cv2.Canny(np.uint8(gray), 125, 200)

    # Obtain contours of result of Canny detection
    # With mode=cv2.RETR_LIST, some images had nested contours with more than 4
    # total points; using RETR_EXTERNAL instead makes sure only the outermost
    # contour is considered for each possible contour
    cnts, hier = cv2.findContours(can,
                                  mode=cv2.RETR_EXTERNAL,  # Only give outermost contours
                                  method=cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through possible contours to find
    # the one corresponding with the document
    for cnt in cnts:
        # Get perimeter of current contour
        peri = cv2.arcLength(cnt, closed=True)
        # Approximate closed polygon from current contour
        # with maximum perimeter difference of 1%
        approx = cv2.approxPolyDP(cnt, epsilon=0.01 * peri, closed=True)

        # If this contour has four points, we can
        # assume we've found the document
        if len(approx) == 4:
            break

    # Return corners of contour as a 2D numpy array (resized to match original)
    # NOTE: This might not give exactly 4 corners. I have no idea what happens
    # or why rectification might still work if it doesn't.
    pts = np.array([5 * approx[i][0] for i in range(approx.shape[0])])

    # Make sure orientation of corners is as expected
    if len(pts) >= 4:
        pts = fixOrientation(pts)

    return pts


# Given an image and set of 4 corner points denoting a perspective-warped
# rectangle, computes the actual aspect ratio of the rectangle and returns
# a width/height tuple at the obtained ratio
def getPerspectiveDimensions(img, pts):
    # Principal point (u0, v0) is center of image
    # This doesn't necessarily apply to all cameras, but it does for most
    u0 = img.shape[1] / 2.0
    v0 = img.shape[0] / 2.0

    w1 = euclidean(pts[3], pts[2])
    w2 = euclidean(pts[0], pts[1])

    h1 = euclidean(pts[3], pts[0])
    h2 = euclidean(pts[2], pts[1])

    w = max(w1, w2)
    h = max(h1, h2)

    ar_vis = float(w) / float(h)

    m1 = np.array((pts[3][0], pts[3][1], 1)).astype('float32')
    m2 = np.array((pts[2][0], pts[2][1], 1)).astype('float32')
    m3 = np.array((pts[0][0], pts[0][1], 1)).astype('float32')
    m4 = np.array((pts[1][0], pts[1][1], 1)).astype('float32')

    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21, n22, n23 = n2[:3]
    n31, n32, n33 = n3[:3]

    f = np.sqrt(
        np.abs(  # Need abs to avoid sqrt of negative
            (1.0 / (n23*n33))  # Pixels are square so s = 1
            * ((n21*n31 - (n21*n33 + n23*n31)*u0 + n23*n33*u0**2)
               + (n22*n32 - (n22*n33 + n23*n32)*v0 + n23*n33*v0**2))
        )
    )

    A = np.array([[f, 0, u0],
                  [0, f, v0],  # Again s = 1, so s*f = f
                  [0, 0, 1]]).astype('float32')

    At = np.transpose(A)
    # Not particularly fast but meh
    Ati = np.linalg.inv(At)
    Ai = np.linalg.inv(A)

    # Actual aspect ratio
    ar_real = np.sqrt(np.dot(np.dot(np.dot(n2, Ati), Ai), n2)
                      / np.dot(np.dot(np.dot(n3, Ati), Ai), n3))

    if ar_real < ar_vis:
        new_w = int(w)
        new_h = int(new_w / ar_real)
    else:
        new_h = int(h)
        new_w = int(ar_real * new_h)

    return (new_w, new_h)


# Computes the homography matrix needed to convert the
# four points in im1_points to those in im2_pts
def computeH(im1_pts, im2_pts):
    A = []

    # Build matrix A using points from images, b gets zeroes
    for i in range(4):
        A.append([-im1_pts[i, 0], -im1_pts[i, 1], -1,
                  0, 0, 0,
                  im1_pts[i, 0] * im2_pts[i, 0], im1_pts[i, 1] * im2_pts[i, 0], im2_pts[i, 0]])
        A.append([0, 0, 0,
                  -im1_pts[i, 0], -im1_pts[i, 1], -1,
                  im2_pts[i, 1] * im1_pts[i, 0], im2_pts[i, 1] * im1_pts[i, 1], im2_pts[i, 1]])

    # Build b and append 9th row/value
    b = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    A.append(b)

    # Solve for homography using least squares solution
    H = np.resize(np.linalg.lstsq(A, b, rcond=None)[0], (3, 3))
    return H


# Given an image containing a rectangular document, performs homography
# warping to rectify document into upright, forward-facing rectangle
def rectify(img):
    print('\nWould you like to input corner coordinates manually? (enter \'Y\' for manual input, or enter anything else for automatic corner detection):')

    choice = str(input())

    if len(choice) == 1 and choice.lower() == 'y':
        # Get coordinates of corners through console input

        pts = []

        # No input validation happens here, please put proper values :)
        print('Enter x- and y-coordinate of top-left corner, separated by a space:')
        coords = input().split()
        pts.append([int(coords[0]), int(coords[1])])

        print('Enter x- and y-coordinate of top-right corner:')
        coords = input().split()
        pts.append([int(coords[0]), int(coords[1])])

        print('Enter x- and y-coordinate of bottom-right corner:')
        coords = input().split()
        pts.append([int(coords[0]), int(coords[1])])

        print('Enter x- and y-coordinate of bottom-left corner:')
        coords = input().split()
        pts.append([int(coords[0]), int(coords[1])])

        pts = np.array(pts)
    else:
        # Otherwise, automatically detect document corners
        pts = detectCorners(img)
        if len(pts) < 4:
            return None

    # Get width and height of rectified document rectangle
    w, h = getPerspectiveDimensions(img, pts)

    # Make sure document is upright (might still be upside-down but oh well)
    if w > h:
        # Need to "rotate" corner points around by 90 degrees
        tmp_pt = pts[0].copy()
        pts[0] = pts[3].copy()
        pts[3] = pts[2].copy()
        pts[2] = pts[1].copy()
        pts[1] = tmp_pt

        # And swap width/height
        temp = w
        w = h
        h = temp

    # Set up new rectangular points
    new_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    # Compute homography matrix
    H = computeH(pts, new_pts)
    # Apply homography
    warp = cv2.warpPerspective(img, H, (w, h))

    return warp


# List of image URLs to run the method on
# Feel free to un-comment or add some, I guess
urls = [
    'https://i.imgur.com/KVOxPCT.jpg',      # Sheet music 1
    # Sheet music 2 (less extreme angle)
    'https://i.imgur.com/YH7HVC0.jpg',
    'https://i.imgur.com/OoOSEDY.jpg',      # Sheet music 2, rotated 90deg clockwise
    'https://i.redd.it/bebi64i3kbb31.jpg',  # Channel list
]

# Base size for image display (large images are tough to see in tiny console)
# Set to 0 for original-size images
size = 500

for i, url in enumerate(urls):
    print('Image ', i + 1, ':', sep='')

    # Read image, store original in case img is modified
    img = np.float32(imread(url))
    og = img.copy()

    # Obtain rectified image
    res = rectify(img)
    # If there was an error (probably no document detected), skip
    if res is None:
        print('No document detected in image.')
        continue

    # Display original
    print('\nOriginal:')
    og_bgr = cv2.cvtColor(og, cv2.COLOR_RGB2BGR)
    if size == 0:
        plt.imshow(cv2.cvtColor(og, cv2.COLOR_RGB2BGR))
    else:
        ratio = float(og.shape[1]) / float(og.shape[0])
        plt.imshow(cv2.resize(og_bgr, (int(ratio * size), size)))

    plt.show()

    # Display result
    print('\nRectified:')
    res_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    if size == 0:
        plt.imshow(res_bgr)
    else:
        watio = float(res.shape[1]) / float(res.shape[0])
        plt.imshow(cv2.resize(res_bgr, (int(watio * size), size)))

    plt.show()

    print('\n\n')
