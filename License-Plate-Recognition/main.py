import argparse
from pathlib import Path
import cv2
import time
import os
from src.lp_recognition import E2E

# Define a function to get arguments
def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_folder', help='Path to the folder containing images', default='./samples/')
    arg.add_argument('-o', '--output_folder', help='Path to the output folder', default='./output/')
    return arg.parse_args()

# Load arguments
args = get_arguments()
folder_path = Path(args.image_folder)
output_folder = Path(args.output_folder)

# Create output folder if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

# Initialize the model
model = E2E()

# Start timer
start = time.time()

count = 0
# Process each image in the folder
for img_path in folder_path.glob("*.jpg"):  # You can add other image formats if needed, e.g., "*.png"
    print(f"Processing image: {img_path}")
    count += 1

    # Read image
    img = cv2.imread(str(img_path))


    # Recognize license plate
    result_image = model.predict(img)



    # Define output path for the processed image
    output_path = output_folder / img_path.name

    # Save the processed image
    cv2.imwrite(str(output_path), result_image)

# End timer
end = time.time()
# Total time taken
print(f"Total time taken: {end - start:.2f} seconds")
# Frames per second
print(f"Frames per second: {count / (end - start):.2f} fps")