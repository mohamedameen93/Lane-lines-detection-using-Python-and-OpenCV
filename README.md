# **Lane Lines Detection Using Python and OpenCV** 
In this project, I used Python and OpenCV to detect lane lines on the road. # I developed a processing pipeline that works on a series of individual images, and applied the result to a video stream.

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Pipeline architecture:
---
1. Load test images.
2. Apply Color Selection
3. Apply Canny edge detection.
   - Apply gray scaling to the images.
   - Apply Gaussian smoothing.
   - Perform Canny edge detection.
4. Determine the region of interest.
5. Apply Hough transform.
6. Average and extrapolating the lane lines.
7. Apply on video streams.
I'll explain each step in details below.

Environement:
---
- Windows 7
- Anaconda 4.3.29
- Python 3.6.2
- OpenCV 3.1.0

