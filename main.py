import os
import datetime
from ultralytics import YOLO
import cv2
from helper import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque

CONFIDENCE_THRESHOLD = 0.30
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Directory to save images
SAVE_DIR = 'saved_images'
os.makedirs(SAVE_DIR, exist_ok=True)

video_cap = cv2.VideoCapture("test_v006.mp4")
writer = create_video_writer(video_cap, "output.mp4")

model = YOLO("config/model/detector_yolov8s_best_v0002.pt")
tracker = DeepSort(max_age=25, max_cosine_distance=0.35, max_iou_distance=0.3)

# Track saved IDs to avoid saving duplicates
saved_ids = set()

# Dictionary to track votes for each track ID
track_votes = {}

# Maximum number of frames for voting window
VOTING_WINDOW_SIZE = 5

while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    if not ret:
        break
    detections = model(frame, conf=0.20)[0]
    results = []

    ############
    # DETECTION
    ############

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # confidence (i.e., probability) associated with the prediction
        confidence = data[4]

        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # bounding box coordinates
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        width = xmax - xmin
        height = ymax - ymin
        area = width * height

        if area < 50:
            # print(" AREA IS: ", area) # For Debugging
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

        width = xmax - xmin
        height = ymax - ymin
        area = width * height


        class_id = int(data[5])
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    ##########
    # TRACKING
    ##########


    # update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # Update the track votes for this track_id
        if track_id not in track_votes:
            track_votes[track_id] = deque(maxlen=VOTING_WINDOW_SIZE)

        # Add the current class id to the track's votes
        track_votes[track_id].append(class_id)

        # Determine the majority vote label if there are enough frames (VOTING_WINDOW_SIZE)
        if len(track_votes[track_id]) == VOTING_WINDOW_SIZE:
            majority_class_id = max(set(track_votes[track_id]), key=track_votes[track_id].count)
        else:
            majority_class_id = class_id  # If not enough frames, use current class_id

        # If it's a new track ID, save the image
        if track_id not in saved_ids:
            # Mark this track ID as saved
            saved_ids.add(track_id)

            # Save the full image (frame) with a unique name
            full_image_path = os.path.join(SAVE_DIR, f"full_image_{track_id}.jpg")
            cv2.imwrite(full_image_path, frame)

            # Crop the bounding box and save the cropped image
            cropped_image = frame[ymin:ymax, xmin:xmax]
            cropped_image_path = os.path.join(SAVE_DIR, f"cropped_image_{track_id}.jpg")
            cv2.imwrite(cropped_image_path, cropped_image)

        # Draw the bounding box and the track ID
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # Display the majority class label
        cv2.putText(frame, f"Label: {majority_class_id}", (xmin, ymin - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)


    end = datetime.datetime.now()
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)


    cv2.imshow("Frame", frame)
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()
