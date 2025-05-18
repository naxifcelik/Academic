import pyautogui
import numpy as np
import time
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Initialize the model
base_model = VGG16(weights='imagenet', include_top=False)
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)

# Grid tracking variables
visited_grids = defaultdict(int)
current_grid = None
face_corner_region = (1425, 715, 1740, 1030)  # Adjust based on your screenshot (bottom right 315x315)

# Store reference facial expressions
normal_face_features = None
shocked_face_features = None


def get_screenshot():
    """Capture the current game screen"""
    screenshot = pyautogui.screenshot()
    return np.array(screenshot)


def extract_face_features(img):
    """Extract features from the face region using the CNN"""
    face_img = img[face_corner_region[1]:face_corner_region[3],
               face_corner_region[0]:face_corner_region[2]]

    # Resize to VGG16 input size
    face_img = cv2.resize(face_img, (224, 224))
    face_img = image.img_to_array(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    face_img = preprocess_input(face_img)

    # Extract features
    features = feature_extractor.predict(face_img)
    features = features.flatten()

    return features


def detect_grid_transition(img):
    """Detect when the character moves to a new grid"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Use Hough Line Transform to detect grid lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Detect grid position based on character position relative to grid lines
    # This is a simplification - you'd need to adjust for your specific game
    if lines is not None:
        # Process line intersections to determine grid position
        character_center_x = img.shape[1] // 2
        character_center_y = img.shape[0] // 2

        # Find closest horizontal and vertical lines
        # Use these to determine grid position
        grid_x, grid_y = estimate_grid_position(lines, character_center_x, character_center_y)
        return (grid_x, grid_y)
    return None


def estimate_grid_position(lines, char_x, char_y):
    """Estimate which grid cell the character is in based on detected lines"""
    # This is a simplified implementation
    # You might need to adjust this based on your game's grid size and layout
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < abs(x2 - x1):  # Horizontal line
            horizontal_lines.append((y1 + y2) // 2)
        else:  # Vertical line
            vertical_lines.append((x1 + x2) // 2)

    horizontal_lines.sort()
    vertical_lines.sort()

    # Determine grid position
    grid_x = sum(1 for x in vertical_lines if x < char_x)
    grid_y = sum(1 for y in horizontal_lines if y < char_y)

    return grid_x, grid_y


def is_shocked_face(features):
    """Determine if the face shows a shocked expression"""
    global normal_face_features, shocked_face_features

    # If we don't have reference features yet, we can't determine
    if normal_face_features is None:
        # Assume the first face we see is normal
        normal_face_features = features
        return False

    # Compare with normal face using cosine similarity
    similarity = cosine_similarity(features.reshape(1, -1),
                                   normal_face_features.reshape(1, -1))[0][0]

    # If similarity is low, this might be a shocked face
    if similarity < 0.8:  # Threshold to adjust based on testing
        if shocked_face_features is None:
            shocked_face_features = features
        return True
    return False


def choose_next_move(current_grid, visited_grids):
    """Choose the next direction to move based on visited grids"""
    # Direction priorities: right, down, up, left
    directions = [('right', 'right'), ('down', 'down'), ('up', 'up'), ('left', 'left')]

    # Shuffle to add some randomness when needed
    np.random.shuffle(directions)

    # Sort by visit count (prefer less visited directions)
    directions.sort(key=lambda d: visited_grids[get_next_grid(current_grid, d[0])])

    return directions[0][1]  # Return the key to press


def get_next_grid(current_grid, direction):
    """Calculate the next grid position based on direction"""
    x, y = current_grid
    if direction == 'right':
        return (x + 1, y)
    elif direction == 'left':
        return (x - 1, y)
    elif direction == 'up':
        return (x, y - 1)
    elif direction == 'down':
        return (x, y + 1)
    return current_grid


def move_character(direction, run=False):
    """Move the character in the specified direction"""
    # Optional: Hold shift to run
    if run:
        pyautogui.keyDown('shift')

    # Press the arrow key
    pyautogui.keyDown(direction)
    time.sleep(0.2)  # Adjust timing based on game speed
    pyautogui.keyUp(direction)

    if run:
        pyautogui.keyUp('shift')

    # Give the game time to respond
    time.sleep(0.1)


def main():
    global current_grid

    print("Starting navigation in 3 seconds...")
    time.sleep(3)  # Give time to switch to the game window

    # Initialize current grid
    screenshot = get_screenshot()
    current_grid = detect_grid_transition(screenshot)
    if current_grid is None:
        current_grid = (0, 0)  # Start at origin if we can't detect

    visited_grids[current_grid] = 1

    # Main navigation loop
    while True:
        screenshot = get_screenshot()
        face_features = extract_face_features(screenshot)

        # Detect if we're near a mine
        danger = is_shocked_face(face_features)

        # Update grid position
        new_grid = detect_grid_transition(screenshot)
        if new_grid is not None and new_grid != current_grid:
            current_grid = new_grid
            visited_grids[current_grid] += 1

        # Choose next move
        if danger:
            # If in danger, try to backtrack
            last_direction = choose_next_move(current_grid, visited_grids)
            move_character(last_direction, run=True)  # Run away quickly
        else:
            # Normal exploration
            next_direction = choose_next_move(current_grid, visited_grids)
            move_character(next_direction)

        # Optional: Add a way to exit the loop
        if pyautogui.position()[0] < 10:  # Move mouse to left edge to stop
            break


if __name__ == "__main__":
    main()