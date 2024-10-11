# import cv2
# import os

# # Initialize the ORB detector with more features
# orb = cv2.ORB_create(nfeatures=1000)

# # Function to load dataset images and extract ORB descriptors
# def load_dataset(dataset_path):
#     dataset = []
#     for filename in os.listdir(dataset_path):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             img = cv2.imread(os.path.join(dataset_path, filename), 0)
#             if img is not None:  # Ensure the image was loaded
#                 keypoints, descriptors = orb.detectAndCompute(img, None)
#                 if descriptors is not None:
#                     print(f"Descriptors found in the dataset image '{filename}': {descriptors.shape[0]}")  # Debugging
#                     dataset.append((filename, descriptors))
#                 else:
#                     print(f"Warning: No descriptors found for image {filename}")
#             else:
#                 print(f"Error: Could not load image {filename}")
#     return dataset

# # Load the dataset (make sure to unzip the dataset first if needed)
# dataset_path = 'dataset-resized/dataset-resized/trash'  # Update with your dataset folder path
# dataset = load_dataset(dataset_path)

# if len(dataset) == 0:
#     print("Error: No valid descriptors found in the dataset.")
#     exit()

# # Set a higher similarity threshold for the minimum number of good matches
# min_good_matches = 30  # Increase the number of good matches required

# # Open the camera feed
# cap = cv2.VideoCapture("http://192.168.1.2:8080/video")

# if not cap.isOpened():
#     print("Error: Could not open the camera.")
#     exit()

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("Failed to capture image")
#         break

#     # Convert to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian blur to reduce noise (optional)
#     gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

#     # Detect keypoints and compute descriptors for the current frame
#     keypoints_frame, descriptors_frame = orb.detectAndCompute(gray_frame, None)

#     if descriptors_frame is None:
#         print("No descriptors found in the camera frame.")
#         continue
#     else:
#         print(f"Descriptors found in the camera frame: {descriptors_frame.shape[0]}")  # Debugging

#     # Initialize variables for the best match
#     best_match = None
#     best_num_matches = 0

#     # Compare the frame descriptors with each image in the dataset
#     for image_name, descriptors_dataset in dataset:
#         bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#         matches = bf.knnMatch(descriptors_frame, descriptors_dataset, k=2)

#         # Apply ratio test as per Lowe's paper
#         good_matches = []
#         for match in matches:
#             if len(match) == 2:
#                 m, n = match
#                 if m.distance < 0.75 * n.distance:  # Modify the ratio test
#                     good_matches.append(m)

#         # Only consider matches if they exceed the minimum threshold
#         if len(good_matches) > best_num_matches and len(good_matches) > min_good_matches:
#             best_num_matches = len(good_matches)
#             best_match = image_name

#     # Display the result only if a valid match is found
#     if best_match is not None:
#         cv2.putText(frame, f"Detected: {best_match} ({best_num_matches} matches)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#         print(f"Object detected: {best_match}, Good matches: {best_num_matches}")
#     else:
#         print("No valid match found.")

#     # Show the camera frame with results
#     cv2.imshow('Waste Material Detection', frame)

#     # Break loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close the window
# cap.release()
# cv2.destroyAllWindows()



import cv2
import os

# Initialize the ORB detector with more features
orb = cv2.ORB_create(nfeatures=1000)

# Function to load dataset images and extract ORB descriptors
def load_dataset(dataset_path):
    dataset = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(dataset_path, filename), 0)
            if img is not None:  # Ensure the image was loaded
                keypoints, descriptors = orb.detectAndCompute(img, None)
                if descriptors is not None and len(descriptors) > 0:  # Add check for empty descriptors
                    print(f"Descriptors found in the dataset image '{filename}': {descriptors.shape[0]}")  # Debugging
                    dataset.append((filename, descriptors))
                else:
                    print(f"Warning: No descriptors found for image {filename}")
            else:
                print(f"Error: Could not load image {filename}")
    return dataset

# Function to match descriptors using BFMatcher
def match_features(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        return []  # Return empty list if any descriptors are None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Apply knnMatch, ensuring that descriptors have enough points
    if len(descriptors1) >= 2 and len(descriptors2) >= 2:  # Check that there are at least 2 descriptors
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for match in matches:
            if len(match) == 2:  # Ensure there are two matches
                m, n = match
                if m.distance < 0.75 * n.distance:  # Apply ratio test
                    good_matches.append(m)
        return good_matches
    else:
        print("Not enough descriptors to match.")
        return []

# Load the dataset (make sure to unzip the dataset first if needed)
dataset_path = 'dataset-resized/dataset-resized/trash'  # Update with your dataset folder path
dataset = load_dataset(dataset_path)

if len(dataset) == 0:
    print("Error: No valid descriptors found in the dataset.")
    exit()

# Set a higher similarity threshold for the minimum number of good matches
min_good_matches = 30  # Increase the number of good matches required

# Open the camera feed
cap = cv2.VideoCapture("http://192.168.1.2:8080/video")

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

    # Apply Gaussian blur to reduce noise (optional)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Detect keypoints and compute descriptors for the current frame
    keypoints_frame, descriptors_frame = orb.detectAndCompute(gray_frame, None)

    if descriptors_frame is None or len(descriptors_frame) == 0:
        print("No descriptors found in the camera frame.")
        continue
    else:
        print(f"Descriptors found in the camera frame: {descriptors_frame.shape[0]}")  # Debugging

    # Initialize variables for the best match
    best_match = None
    best_num_matches = 0

    # Compare the frame descriptors with each image in the dataset
    for image_name, descriptors_dataset in dataset:
        good_matches = match_features(descriptors_frame, descriptors_dataset)

        # Only consider matches if they exceed the minimum threshold
        if len(good_matches) > best_num_matches:
            best_num_matches = len(good_matches)
            best_match = image_name

    # Display the result only if a valid match is found
    if best_match is not None and best_num_matches >= min_good_matches:
        # Display best match and number of good matches as similarity score
        cv2.putText(frame, f"Detected: {best_match}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Similarity Score: {best_num_matches}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        print(f"Object detected: {best_match}, Good matches: {best_num_matches}")
    else:
        print("No valid match found.")

    # Show the camera frame with results
    cv2.imshow('Waste Material Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
