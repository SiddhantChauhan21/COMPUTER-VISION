# EXPERIMENT NO. 8
# Application of Optical Flow for Real-Time Object Tracking and Motion Analysis

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os 

# 1. IMPORT REQUIRED LIBRARIES AND LOAD/CAPTURE VIDEO

VIDEO_PATH = "bird.mp4"       # Change to 0 for webcam

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

ret, first_frame = cap.read()

if not ret:
    print("Error: Unable to read video.")
    cap.release()
    exit()

first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# 2. IDENTIFY MOVING OBJECT / SELECT FEATURE POINTS USING
#    SHI-TOMASI CORNER DETECTION


feature_params = {
    "maxCorners": 100,
    "qualityLevel": 0.3,
    "minDistance": 7,
    "blockSize": 7
}

p0 = cv2.goodFeaturesToTrack(
    first_gray,
    mask=None,
    **feature_params
)

# 3. APPLY LUCAS-KANADE OPTICAL FLOW
#    TO TRACK SELECTED POINTS

lk_params = {
    "winSize": (15, 15),
    "maxLevel": 2,
    "criteria": (
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
}

# 4. DRAW MOTION TRAJECTORIES AND DISPLACEMENT VECTORS

trajectory_mask = np.zeros_like(first_frame)
# 5. CALCULATE DIRECTION AND MAGNITUDE OF OBJECT MOTION
#    AND DISPLAY TRACKING RESULTS IN REAL TIME


prev_gray = first_gray.copy()

total_displacement = 0
frame_count = 0
motion_values = []
direction_values = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Lucas-Kanade Optical Flow

    if p0 is not None and len(p0) > 0:

        p1, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            p0,
            None,
            **lk_params
        )

        if p1 is not None:

            good_new = p1[status == 1]
            good_old = p0[status == 1]

            magnitudes = []
            directions = []

            # 4 & 5. TRAJECTORY, DISPLACEMENT, MAGNITUDE
            #       AND DIRECTION

            for new, old in zip(good_new, good_old):

                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()

                dx = x_new - x_old
                dy = y_new - y_old

                magnitude = np.sqrt(dx ** 2 + dy ** 2)
                direction = np.degrees(np.arctan2(dy, dx))

                magnitudes.append(magnitude)
                directions.append(direction)

                total_displacement += magnitude

                # Draw trajectory
                trajectory_mask = cv2.line(
                    trajectory_mask,
                    (int(x_old), int(y_old)),
                    (int(x_new), int(y_new)),
                    (0, 255, 0),
                    2
                )

                # Draw displacement vector
                frame = cv2.arrowedLine(
                    frame,
                    (int(x_old), int(y_old)),
                    (int(x_new), int(y_new)),
                    (0, 0, 255),
                    2,
                    tipLength=0.3
                )

                # Draw feature point
                frame = cv2.circle(
                    frame,
                    (int(x_new), int(y_new)),
                    4,
                    (255, 0, 0),
                    -1
                )

            # Average motion
            if len(magnitudes) > 0:

                avg_magnitude = np.mean(magnitudes)
                avg_direction = np.mean(directions)

                motion_values.append(avg_magnitude)
                direction_values.append(avg_direction)

                # 8. ESTIMATE OBJECT DISPLACEMENT AND
                #    TRAJECTORY-BASED MOTION PARAMETERS
                

                displacement_x = np.mean(
                    good_new[:, 0] - good_old[:, 0]
                )

                displacement_y = np.mean(
                    good_new[:, 1] - good_old[:, 1]
                )

                velocity = avg_magnitude * fps

                cv2.putText(
                    frame,
                    f"Displacement: {avg_magnitude:.2f} pixels",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Direction: {avg_direction:.2f} deg",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Velocity: {velocity:.2f} pixels/sec",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"dx: {displacement_x:.2f}, dy: {displacement_y:.2f}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

            # Update tracked points
            p0 = good_new.reshape(-1, 1, 2)

        else:
            p0 = None

    # Combine frame and trajectories
    output = cv2.add(frame, trajectory_mask)

    cv2.imshow(
        "Lucas-Kanade Optical Flow",
        output
    )

    # 7. REPEAT EXPERIMENT USING FARNEBACK DENSE OPTICAL FLOW
    #    AND COMPARE TRACKING PERFORMANCE
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    flow_magnitude, flow_angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    # Create HSV representation of dense optical flow
    hsv = np.zeros_like(frame)
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(
        flow_magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    dense_flow = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    cv2.imshow(
        "Farneback Dense Optical Flow",
        dense_flow
    )

    # 6. UPDATE FRAME FOR NEXT ITERATION

    prev_gray = gray.copy()

    frame_count += 1

    # 2. RE-DETECT FEATURES IF TRACKED POINTS ARE LOST

    if p0 is None or len(p0) < 10:

        p0 = cv2.goodFeaturesToTrack(
            gray,
            mask=None,
            **feature_params
        )

        # Start a new trajectory when features are re-detected
        trajectory_mask = np.zeros_like(frame)

    # Press ESC to stop
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 8. MOTION ANALYSIS

if len(motion_values) > 0:

    average_displacement = np.mean(motion_values)
    maximum_displacement = np.max(motion_values)
    average_direction = np.mean(direction_values)

    total_time = frame_count / fps
    average_velocity = average_displacement * fps

    print("\n========== MOTION ANALYSIS ==========")
    print(f"Total Frames         : {frame_count}")
    print(f"Video FPS             : {fps:.2f}")
    print(f"Total Time            : {total_time:.2f} seconds")
    print(f"Average Displacement  : {average_displacement:.2f} pixels/frame")
    print(f"Maximum Displacement  : {maximum_displacement:.2f} pixels/frame")
    print(f"Average Direction     : {average_direction:.2f} degrees")
    print(f"Average Velocity      : {average_velocity:.2f} pixels/second")

# 9. CLOSE ALL OPENCV WINDOWS AFTER VIDEO ENDS

cap.release()

cv2.destroyAllWindows()

# Give OpenCV time to process window closing
cv2.waitKey(1)

# 10. DISPLAY MOTION GRAPHS


if len(motion_values) > 0:

    time_axis = np.arange(len(motion_values)) / fps

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_axis, motion_values, color="blue")
    plt.title("Object Motion Magnitude")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Magnitude (pixels/frame)")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time_axis, direction_values, color="red")
    plt.title("Object Motion Direction")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Direction (degrees)")
    plt.grid()

    plt.tight_layout()
    plt.show()

# PROGRAM FINISHED


print("\nProgram completed successfully.")
