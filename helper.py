import cv2

import cv2

def create_video_writer(video_cap, output_filename):


    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)

    # Ensure width and height are even, as many codecs require this
    if frame_width % 2 != 0:
        frame_width += 1
    if frame_height % 2 != 0:
        frame_height += 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer
