import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Import the required libraries
# cv2, numpy and matplotlib are imported above

# Step 2: Load the image and convert it into grayscale
image = cv2.imread("bird.jpeg")

if image is None:
    print("Error: Image not found.")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Display original image and its histogram
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Original Grayscale Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.hist(gray.ravel(), bins=256, range=[0, 256], color='black')
    plt.title("Histogram of Original Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Step 4 & 5: Apply Histogram Equalization
    equalized = cv2.equalizeHist(gray)

    # Display histogram equalized image
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(equalized, cmap='gray')
    plt.title("Histogram Equalized Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.hist(equalized.ravel(), bins=256, range=[0, 256], color='black')
    plt.title("Histogram After Equalization")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Step 6: Apply CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_image = clahe.apply(gray)

    # Step 7: Display CLAHE image and histogram
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(clahe_image, cmap='gray')
    plt.title("CLAHE Enhanced Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.hist(clahe_image.ravel(), bins=256, range=[0, 256], color='black')
    plt.title("Histogram After CLAHE")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Step 8: Compare original, histogram equalized and CLAHE images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(equalized, cmap='gray')
    plt.title("Histogram Equalization")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(clahe_image, cmap='gray')
    plt.title("CLAHE")
    plt.axis("off")

    plt.tight_layout()
    plt.show()