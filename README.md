# Project Report
(featuring lots of broken LateX!)

<br>

<ins>Image warping</ins>

To obtain the desired (rectified) set of corners, I first employed a naïve
approach in which I obtained the width and height of the polygon with
vertices at the document corners (detection of these corners is
discussed later), and then used these as the width and height of the desired
image by defining the transformed corners as $(0, 0)$, $(0, h)$, $(w, h)$, and
$(w, 0)$, in proper corner order (top-left, top-right, bottom-right,
bottom-left).

This method does not work well for photos taken at
shallow angles, as the dimension(s) parallel to the focal length will be
significantly shrunken and so the aspect ratio will be different from the
document's actual aspect ratio. However, since we can assume the document is a
rectangle, we can estimate the focal length of the camera and the actual
aspect ratio. I won't pretend to fully understand the math behind it, but the
general method I followed can be found in https://www.microsoft.com/en-us/research/uploads/prod/2016/11/Digital-Signal-Processing.pdf. This process is
implemented in the `getPerspectiveDimensions` function.

For the `computeH` function, I simply followed the method from the slides.
I first build the appropriate matrices for the equation

<p align="center">
    [insert fancy equation here]
</p>

where $(x_i, y_i)$ is the coordinate of the $i$th document corner from the
original image and $(x_i', y_i')$ is a rectified corner
(from the desired image). Then, using a numpy function, I obtain a
least-squares solution to this equation for $h_i$, the elements of the
desired 3 $\times$ 3 homography matrix. This is very similar to the function
shown in class, with minor adjustments because that function didn't work
properly in my case.

To apply the obtained homography to the image, I'm simply using OpenCV's
`warpPerspective` function, as I could not get a homebrewed function working
for this part. This function takes an image, a homography (perspective warp)
matrix, and a size for the resultant image. Using the proper parameters
(including the width/height obtained from the original document image),
I apply the homography I computed using `computeH` to the original image to
obtain the desired, rectified image.

<br>


<ins>Corner detection</ins>

The corner detection method I'm using is based on the code found at
https://github.com/adityaguptai/Document-Boundary-Detection.
The general approach for my `detectCorners` function is:

1. Apply a Gaussian blur to grayscale image (doesn't have to be Gaussian,
but it worked the best for me).
2. Perform Canny edge detection on smoothed image. Smoothing the image
beforehand results in less noise being detected and makes the document edges
more likely to be the only edges detected.
3. Detect the possible contours of Canny result. Again, I won't pretend to fully
understand the mechanics behind OpenCV's `findContours` function. As I currently
understand it, this function gives a list of possible sets of bounding points
for objects in the image.
4. For each contour, approximate a polygon from its points. If the estimated
shape has exactly four points, then we can assume we've detected the document
(which, of course, has four corners). The coordinates of the points that define
this polygon are the coordinates of the document corners.
5. Orient these corners properly (top-left, top-right, bottom-right,
bottom-left, in that order). This is the purpose of the `fixOrientation`
function.

The exact behavior of the program in the case where there are no contour
polygons with exactly four points is unknown to me. Certainly one way to
resolve this is to simply not attempt to rectify the document in this case;
this is how I handled the case where there are less than 4 points.
However, in some cases where there were more than four points returned,
the algorithm still worked, since the first four points in the array of corners
happened to still denote the corners of the document.

<br>

<ins>Conclusions</ins>

This program surprises me with how well it works. Although it doesn't work
perfectly with photos taken at very shallow angles, this is to be at least
somewhat expected; best results will obviously be achieved the closer to
straight-forward the photo is taken.

One area in which this program can be improved is the corner detection.
For example, currently, a hardcoded Gaussian kernel size is used when
smoothing the image prior to Canny edge detection. However,
the current size (11 $\times$ 11) doesn't work for large images, because
the gradient across just 11 square pixels of a very large image will be
relatively small, so the image won't be smoothed enough for Canny detection
to avoid detecting noise. An appropriate kernel size could be automatically
computed based on the size of the image, though I did not look into doing so.
