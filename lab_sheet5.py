import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Load image
image = cv2.imread("mountain.webp")

# Check whether image is loaded
if image is None:
    print("Error: Image not found.")
else:
    print("Image loaded successfully.")

# Convert BGR to RGB for displaying
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 5))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()


# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

print("Number of SIFT keypoints:", len(keypoints))
print("SIFT descriptor shape:", descriptors.shape)


sift_image = cv2.drawKeypoints(
    gray,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(10, 7))
plt.imshow(sift_image, cmap="gray")
plt.title("SIFT Keypoints")
plt.axis("off")
plt.show()


# Convert image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Calculate HOG features
hog_features, hog_image = hog(
    image_rgb,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    channel_axis=-1
)

print("HOG feature vector length:", len(hog_features))


# Improve visualization
hog_image_rescaled = exposure.rescale_intensity(
    hog_image,
    in_range=(0, 10)
)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hog_image_rescaled, cmap="gray")
plt.title("HOG Visualization")
plt.axis("off")

plt.tight_layout()
plt.show()



# Load second image
image2 = cv2.imread("image2.jpg")

gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect SIFT features
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Create Brute Force matcher
bf = cv2.BFMatcher()

# Find two nearest matches
matches = bf.knnMatch(descriptors, descriptors2, k=2)

# Apply Lowe's ratio test
good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Total matches:", len(matches))
print("Good matches:", len(good_matches))


matched_image = cv2.drawMatches(
    image,
    keypoints,
    image2,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(14, 7))
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.title("SIFT Feature Matching")
plt.axis("off")
plt.show()
