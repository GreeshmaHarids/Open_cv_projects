import cv2
import random
import time
import mediapipe as mp
import numpy as np

def classify_hand_gesture(landmarks):
    # Define rules for gesture classification based on landmark positions
    # Stone: All fingers are folded (closed fist)
    # Paper: All fingers are extended
    # Scissors: Only the index and middle fingers are extended
    
    # Extracting landmarks for fingertips
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Extracting landmarks for finger bases (proximal phalanges)
    index_base = landmarks[5]
    middle_base = landmarks[9]
    ring_base = landmarks[13]
    pinky_base = landmarks[17]

    # Calculate distances from tips to bases
    distance_thumb = np.linalg.norm(np.array(thumb_tip) - np.array(landmarks[3]))
    distance_index = np.linalg.norm(np.array(index_tip) - np.array(index_base))
    distance_middle = np.linalg.norm(np.array(middle_tip) - np.array(middle_base))
    distance_ring = np.linalg.norm(np.array(ring_tip) - np.array(ring_base))
    distance_pinky = np.linalg.norm(np.array(pinky_tip) - np.array(pinky_base))

    # Define thresholds to determine if fingers are folded or extended
    threshold = 0.1

    if (distance_thumb < threshold and distance_index < threshold and
        distance_middle < threshold and distance_ring < threshold and
        distance_pinky < threshold):
        return "Stone"
    elif (distance_index > threshold and distance_middle > threshold and
          distance_ring > threshold and distance_pinky > threshold):
        return "Paper"
    elif (distance_index > threshold and distance_middle > threshold and
          distance_ring < threshold and distance_pinky < threshold):
        return "Scissors"
    else:
        return "Unknown Gesture"

def determine_game_outcome(random_gesture, user_gesture):
    outcomes = {
        ("Stone", "Stone"): "Same",
        ("Stone", "Paper"): "You Won!",
        ("Stone", "Scissors"): "You Lost!",
        ("Paper", "Stone"): "You Lost!",
        ("Paper", "Paper"): "Same",
        ("Paper", "Scissors"): "You Won!",
        ("Scissors", "Stone"): "You Won!",
        ("Scissors", "Paper"): "You Lost!",
        ("Scissors", "Scissors"): "Same",
    }
    if (random_gesture, user_gesture) in outcomes:
        return outcomes[(random_gesture, user_gesture)]
    else:
        return "Unknown Outcome"

def display_random_image_on_webcam():
    # Initialize random seed
    random.seed(time.time())

    # List of image file paths
    paper_im = "paper.png"
    stone_im = "rock.png"
    scissor_im = "scissors.png"
    images = [
        paper_im,
        stone_im,
        scissor_im
    ]

    # Choose a random image
    random_image = random.choice(images)

    # Determine gesture for the random image
    if stone_im in random_image:
        random_gesture = "Stone"
    elif paper_im in random_image:
        random_gesture = "Paper"
    elif scissor_im in random_image:
        random_gesture = "Scissors"
    else:
        random_gesture = "Unknown Gesture"

    # Load the chosen image
    image = cv2.imread(random_image)

    # Check if the image was loaded successfully
    if image is None:
        print("Could not open or find the image")
        return

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Get the default webcam frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize the image to half of the webcam frame size
    new_width = 1280 // 2
    new_height =  650
    resized_image = cv2.resize(image, (new_width, new_height))

    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Get the start time
    start_time = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)

        if not ret:
            print("Failed to grab frame")
            break

        # Get the current time
        current_time = time.time()

        # Determine which text to display based on time intervals
        if current_time - start_time < 1:
            display_text = "Stone"
        elif current_time - start_time < 2:
            display_text = "Paper"
        elif current_time - start_time < 3:
            display_text = "Scissors"
        else:
            display_text = ""

        # Display the text in the middle of the frame with background rectangle
        if display_text:
            text_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
            text_x = (frame_width - text_size[0]) // 2
            text_y = (frame_height + text_size[1]) // 2

            # Draw a filled rectangle behind the text
            cv2.rectangle(frame, (0, 0),
                          (frame_width, frame_height),
                          (0, 0, 0), -1)  # Background rectangle

            # Display the text
            cv2.putText(frame, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (57, 224, 255), 3, cv2.LINE_AA)

        # If 4 seconds have passed, overlay the image
        if current_time - start_time > 4:
            # Overlay the resized image on the top-left corner of the webcam feed
            frame[0:new_height, 0:new_width] = resized_image

            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame and detect hands
            result = hands.process(frame_rgb)

            # Draw hand landmarks and classify gestures if hands are detected
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Extract landmark positions
                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                    # Classify hand gesture
                    user_gesture = classify_hand_gesture(landmarks)

                    # Draw background rectangles and display the gestures and outcomes
                    cv2.rectangle(frame, (10, frame_height - 130), (frame_width - 10, frame_height - 50), (0, 0, 0), -1)  # Background rectangle for gestures
                    cv2.putText(frame, f"Your Gesture: {user_gesture}", (20, frame_height - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.rectangle(frame, (10, frame_height - 50), (frame_width - 10, frame_height - 10), (0, 0, 0), -1)  # Background rectangle for random gesture
                    cv2.putText(frame, f"Random Gesture: {random_gesture}", (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Determine game outcome
                    game_outcome = determine_game_outcome(random_gesture, user_gesture)

                    cv2.rectangle(frame, (10, 85), (frame_width - 10,0), (0, 0, 0), -1)  # Background rectangle for game outcome
                    cv2.putText(frame, game_outcome, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (90, 245, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Game: Stone Paper Scissor', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close the windows
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    display_random_image_on_webcam()
    display_random_image_on_webcam()
    display_random_image_on_webcam()
