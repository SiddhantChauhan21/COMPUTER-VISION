import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
# TASK 1:
# Import the required Python libraries and load a video sequence

VIDEO_SOURCE = "bird.mp4"

# Check whether the video file exists
if not os.path.exists(VIDEO_SOURCE):
    print("ERROR: Video file not found!")
    print("Make sure 'bird.mp4' is in the same folder as this Python file.")
    exit()

# Open the video
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("ERROR: Could not open the video.")
    exit()

print("Video opened successfully!")

# Get video information

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video Information")
print("-------------------------")
print("Width       :", width)
print("Height      :", height)
print("FPS         :", fps)
print("Total Frames:", total_frames)
print("-------------------------")

# QUESTION / TASK 2:
# Read consecutive video frames and convert them to grayscale
# for optical flow computation.
ret, old_frame = cap.read()

if not ret:
    print("ERROR: Could not read the first frame.")
    cap.release()
    exit()

# Convert first frame to grayscale
old_gray = cv2.cvtColor(
    old_frame,
    cv2.COLOR_BGR2GRAY
)
# TASK 3:
# Implement the Lucas-Kanade Optical Flow algorithm to estimate
# the motion of selected feature points across consecutive frames.


# Shi-Tomasi corner detection parameters
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Lucas-Kanade parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

# Detect feature points in first frame
p0 = cv2.goodFeaturesToTrack(
    old_gray,
    mask=None,
    **feature_params
)

# Mask for drawing trajectories
lk_mask = np.zeros_like(old_frame)

# Store values for analysis

lk_times = []
fb_times = []

# Average motion per frame
lk_motion_per_frame = []
fb_motion_per_frame = []

frame_count = 0

# QUESTION / TASK 5:
# Implement the Farneback Dense Optical Flow algorithm to estimate motion for every pixel in the image.

farneback_params = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

#  TASK 6:
# Display the dense optical flow output using color-coded motion visualization to represent direction and magnitude.
def flow_to_hsv(flow):
    """
    Convert optical-flow vectors to a color image.

    Color/direction represents motion direction.
    Brightness represents motion magnitude.
    """

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    hsv = np.zeros(
        (flow.shape[0], flow.shape[1], 3),
        dtype=np.uint8
    )

    # Motion direction
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Full saturation
    hsv[..., 1] = 255

    # Motion magnitude
    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Convert HSV to BGR for OpenCV display
    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )
# Draw dense optical-flow arrows
def draw_dense_optical_flow(frame, flow, step=25):

    output = frame.copy()

    h, w = flow.shape[:2]

    # Create grid
    y, x = np.mgrid[
        step // 2:h:step,
        step // 2:w:step
    ].astype(int)

    # Draw arrows
    for yy, xx in zip(
        y.flatten(),
        x.flatten()
    ):

        dx, dy = flow[yy, xx]

        start_point = (
            int(xx),
            int(yy)
        )

        end_point = (
            int(xx + dx),
            int(yy + dy)
        )

        cv2.arrowedLine(
            output,
            start_point,
            end_point,
            (0, 255, 0),
            1,
            tipLength=0.3
        )

    return output

# MAIN VIDEO PROCESSING LOOP

while True:

    # Read next frame
    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    frame_count += 1
    # TASK 2: Convert current frame to grayscale
    frame_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )
    # LUCAS-KANADE OPTICAL FLOW

    start_time = time.perf_counter()

    # If feature points are missing, detect new ones
    if p0 is None or len(p0) < 10:

        p0 = cv2.goodFeaturesToTrack(
            old_gray,
            mask=None,
            **feature_params
        )

        # Reset trajectory mask
        lk_mask = np.zeros_like(frame)
    # List to store current frame motion
    current_lk_motion = []
    if p0 is not None:

        # Calculate optical flow
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray,
            frame_gray,
            p0,
            None,
            **lk_params
        )
        if p1 is not None and status is not None:

            # Select successfully tracked points
            good_new = p1[status.ravel() == 1]
            good_old = p0[status.ravel() == 1]
            # TASK 4: Draw trajectories and motion arrows
            for new, old in zip(
                good_new,
                good_old
            ):

                # New point
                a, b = new.ravel()

                # Old point
                c, d = old.ravel()

                a = int(a)
                b = int(b)

                c = int(c)
                d = int(d)


                # Draw trajectory
                lk_mask = cv2.line(
                    lk_mask,
                    (a, b),
                    (c, d),
                    (0, 255, 255),
                    2
                )


                # Draw feature point
                cv2.circle(
                    frame,
                    (a, b),
                    4,
                    (0, 0, 255),
                    -1
                )


                # Draw direction arrow
                cv2.arrowedLine(
                    frame,
                    (c, d),
                    (a, b),
                    (255, 0, 0),
                    1,
                    tipLength=0.3
                )


                # Calculate displacement
                dx = a - c
                dy = b - d

                magnitude = np.sqrt(
                    dx ** 2 + dy ** 2
                )

                current_lk_motion.append(
                    magnitude
                )


            # Update feature points
            if len(good_new) > 0:

                p0 = good_new.reshape(
                    -1,
                    1,
                    2
                )

            else:

                p0 = None


        else:

            p0 = None


    # Calculate Lucas-Kanade processing time
    lk_time = time.perf_counter() - start_time

    lk_times.append(lk_time)


    # Average LK motion for this frame
    if len(current_lk_motion) > 0:

        lk_motion_per_frame.append(
            np.mean(current_lk_motion)
        )

    else:

        lk_motion_per_frame.append(0)


    # Add trajectories
    lk_output = cv2.add(
        frame,
        lk_mask
    )
    # QUESTION / TASK 5:
    # FARNEBACK DENSE OPTICAL FLOW

    start_time = time.perf_counter()

    flow = cv2.calcOpticalFlowFarneback(
        old_gray,
        frame_gray,
        None,
        **farneback_params
    )

    fb_time = time.perf_counter() - start_time

    fb_times.append(fb_time)

    # Calculate Farneback motion magnitude

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    average_fb_motion = np.mean(
        magnitude
    )

    fb_motion_per_frame.append(
        average_fb_motion
    )
    # QUESTION / TASK 6:
    # Color-coded Farneback optical flow

    flow_color = flow_to_hsv(
        flow
    )

    # Draw Farneback motion vectors

    dense_vectors = draw_dense_optical_flow(
        frame,
        flow,
        step=25
    )
    cv2.putText(
        lk_output,
        "Lucas-Kanade Optical Flow",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    cv2.putText(
        flow_color,
        "Farneback Dense Optical Flow",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        dense_vectors,
        "Farneback Motion Vectors",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        lk_output,
        f"Frame: {frame_count}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    display_width = 600

    def resize_frame(img):

        h, w = img.shape[:2]

        scale = display_width / w

        new_h = int(h * scale)

        return cv2.resize(
            img,
            (display_width, new_h)
        )


    lk_display = resize_frame(
        lk_output
    )

    fb_display = resize_frame(
        flow_color
    )

    vector_display = resize_frame(
        dense_vectors
    )

    cv2.imshow(
        "1 - Lucas-Kanade",
        lk_display
    )

    cv2.imshow(
        "2 - Farneback Dense Flow",
        fb_display
    )

    cv2.imshow(
        "3 - Farneback Motion Vectors",
        vector_display
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):

        print("Program stopped by user.")
        break


    old_gray = frame_gray.copy()

cap.release()
cv2.destroyAllWindows()
# QUESTION / TASK 7:
# Compare the performance of Lucas-Kanade and Farneback algorithms in terms of computational complexity and processing speed.

if len(lk_times) > 0:

    avg_lk_time = np.mean(
        lk_times
    )

    avg_lk_fps = 1 / avg_lk_time

else:

    avg_lk_time = 0
    avg_lk_fps = 0


if len(fb_times) > 0:

    avg_fb_time = np.mean(
        fb_times
    )

    avg_fb_fps = 1 / avg_fb_time

else:

    avg_fb_time = 0
    avg_fb_fps = 0


print("\n")
print("=" * 60)
print("OPTICAL FLOW PERFORMANCE")
print("=" * 60)

print(
    f"Lucas-Kanade Average Time : "
    f"{avg_lk_time:.6f} sec/frame"
)

print(
    f"Lucas-Kanade Approx FPS   : "
    f"{avg_lk_fps:.2f}"
)

print()

print(
    f"Farneback Average Time    : "
    f"{avg_fb_time:.6f} sec/frame"
)

print(
    f"Farneback Approx FPS      : "
    f"{avg_fb_fps:.2f}"
)

# QUESTION / TASK 8:
# Analyze the effect of object speed, lighting conditions, and camera motion on optical flow estimation.

print("\n")
print("=" * 60)
print("QUESTION 8 - ANALYSIS")
print("=" * 60)

print("""
1. OBJECT SPEED:
   Faster objects produce larger motion vectors.
   Very fast motion can make feature tracking difficult.

2. LIGHTING CONDITIONS:
   Optical flow works better when brightness remains relatively
   consistent between consecutive frames. Large illumination
   changes can reduce accuracy.

3. CAMERA MOTION:
   Camera movement can produce optical flow throughout the image,
   even when objects themselves are stationary.

4. LUCAS-KANADE:
   Tracks selected feature points and therefore produces sparse
   optical flow.

5. FARNEBACK:
   Estimates motion over the entire image and therefore produces
   dense optical flow.
""")

# QUESTION / TASK 9:
# Evaluate the suitability of each algorithm for applications such
# as object tracking, surveillance, and autonomous navigation.

print("\n")
print("=" * 60)
print("QUESTION 9 - APPLICATIONS")
print("=" * 60)

print("""
Lucas-Kanade:
- Object tracking
- Feature tracking
- Point tracking
- Camera motion estimation
- Real-time applications

Farneback:
- Dense motion estimation
- Motion detection
- Video analysis
- Surveillance
- Scene understanding
- Motion segmentation
""")

# QUESTION / TASK 10:
# Summarize the observations and discuss the significance of
# optical flow in dynamic scene analysis and motion-based
# computer vision systems.

print("\n")
print("=" * 60)
print("QUESTION 10 - CONCLUSION")
print("=" * 60)

print("""
Optical flow is used to estimate apparent motion between
consecutive video frames.

Lucas-Kanade estimates the motion of selected feature points,
whereas Farneback estimates a dense motion field across the
image.

Lucas-Kanade can be useful for feature and object tracking,
while Farneback can be used when motion information across the
image is required.

Optical flow is widely used in computer vision applications such
as object tracking, surveillance, motion detection, autonomous
navigation, video analysis, and dynamic scene understanding.
""")
# PLOT MOTION MAGNITUDE

plt.figure(
    figsize=(10, 5)
)


# Plot Lucas-Kanade
if len(lk_motion_per_frame) > 0:

    plt.plot(
        lk_motion_per_frame,
        label="Lucas-Kanade",
        color="blue"
    )


# Plot Farneback
if len(fb_motion_per_frame) > 0:

    plt.plot(
        fb_motion_per_frame,
        label="Farneback",
        color="red"
    )


plt.title(
    "Optical Flow Motion Magnitude"
)

plt.xlabel(
    "Frame Number"
)

plt.ylabel(
    "Average Motion Magnitude (pixels)"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()  ``

print("\nExperiment completed successfully!")