import cv2
import os

# Initialize the ORB detector
orb = cv2.ORB_create()

# Function to load dataset images and extract ORB descriptors
def load_dataset(dataset_path):
    dataset = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(dataset_path, filename), 0)
            keypoints, descriptors = orb.detectAndCompute(img, None)
            if descriptors is not None:
                dataset.append((filename, descriptors))
    return dataset

# Function to match descriptors using BFMatcher
def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # Apply ratio test as per Lowe's paper
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

# Load the dataset (make sure to unzip the dataset first if needed)
dataset_path = 'dataset-resized\dataset-resized\cardboard'  # Update with your dataset folder path
dataset = load_dataset(dataset_path)

# Set a similarity threshold for the minimum number of good matches
min_good_matches = 10  # You can tune this value

# Open the camera feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors for the current frame
    keypoints_frame, descriptors_frame = orb.detectAndCompute(gray_frame, None)

    # Initialize variables for the best match
    best_match = None
    best_num_matches = 0

    # Compare the frame descriptors with each image in the dataset
    if descriptors_frame is not None:
        for image_name, descriptors_dataset in dataset:
            if descriptors_dataset is not None:
                good_matches = match_features(descriptors_frame, descriptors_dataset)

                # Only consider matches if they exceed the minimum threshold
                if len(good_matches) > best_num_matches and len(good_matches) > min_good_matches:
                    best_num_matches = len(good_matches)
                    best_match = image_name

    # Display the result only if a valid match is found
    if best_match is not None:
        cv2.putText(frame, f"Detected: {best_match}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # *** Draw keypoints on the frame ***
    if keypoints_frame is not None:
        frame_with_keypoints = cv2.drawKeypoints(frame, keypoints_frame, None, (0, 255, 0), cv2.DrawMatchesFlags_DEFAULT)
        cv2.imshow('Waste Material Detection with Keypoints', frame_with_keypoints)
    else:
        # If no keypoints, just show the original frame
        cv2.imshow('Waste Material Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
