import cv2
import mediapipe as mp
import numpy as np
from threading import Thread
from queue import Queue
import time

class ArtisticVisualizer:
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height
        self.queue = Queue(maxsize=1)
        self.current_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.last_direction = None
        
    def create_artistic_visualization(self, direction):
        # Create a blank canvas
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Define colors for different directions
        colors = {
            "Left": (255, 0, 0),    # Blue
            "Right": (0, 255, 0),   # Green
            "Up": (0, 255, 255),    # Yellow
            "Down": (255, 0, 255),  # Purple
            "No hand detected": (128, 128, 128),  # Gray
            "No clear direction": (255, 255, 255)  # White
        }
        
        color = colors.get(direction, (255, 255, 255))
        
        # Create dynamic patterns based on direction
        if direction == "Left":
            # Create leftward flowing pattern
            for i in range(0, self.width, 4):
                cv2.line(image, (i, 0), (i - 100, self.height), color, 2)
        elif direction == "Right":
            # Create rightward flowing pattern
            for i in range(0, self.width, 4):
                cv2.line(image, (i, 0), (i + 100, self.height), color, 2)
        elif direction == "Up":
            # Create upward flowing pattern
            for i in range(0, self.height, 4):
                cv2.line(image, (0, i), (self.width, i - 100), color, 2)
        elif direction == "Down":
            # Create downward flowing pattern
            for i in range(0, self.height, 4):
                cv2.line(image, (0, i), (self.width, i + 100), color, 2)
        else:
            # Create circular pattern for other states
            cv2.circle(image, (self.width//2, self.height//2), 
                      min(self.width, self.height)//3, color, -1)
        
        # Add some dynamic elements
        t = time.time() * 2
        for i in range(8):
            angle = t + i * np.pi / 4
            x = int(self.width/2 + 100 * np.cos(angle))
            y = int(self.height/2 + 100 * np.sin(angle))
            cv2.circle(image, (x, y), 20, (color[0]//2, color[1]//2, color[2]//2), -1)
        
        # Add text
        cv2.putText(image, f"Direction: {direction}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return image

    def update_image(self, direction):
        if direction != self.last_direction and not self.queue.full():
            self.last_direction = direction
            self.queue.put(direction)

    def visualization_loop(self):
        while True:
            if not self.queue.empty():
                direction = self.queue.get()
                self.current_image = self.create_artistic_visualization(direction)
            time.sleep(0.1)

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
        return frame

    def get_direction(self, frame):
        if not self.results.multi_hand_landmarks:
            return "No hand detected"

        hand_landmarks = self.results.multi_hand_landmarks[0]
        
        wrist = hand_landmarks.landmark[0]
        index_finger_tip = hand_landmarks.landmark[8]
        
        index_vector = np.array([
            index_finger_tip.x - wrist.x,
            index_finger_tip.y - wrist.y
        ])
        
        magnitude = np.sqrt(index_vector[0]**2 + index_vector[1]**2)
        if magnitude == 0:
            return "No clear direction"
            
        index_vector = index_vector / magnitude
        
        if abs(index_vector[0]) > abs(index_vector[1]):
            return "Right" if index_vector[0] > 0 else "Left"
        else:
            return "Down" if index_vector[1] > 0 else "Up"

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    visualizer = ArtisticVisualizer()
    
    # Start the visualization thread
    viz_thread = Thread(target=visualizer.visualization_loop, daemon=True)
    viz_thread.start()
    
    # Create windows
    cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Artistic Visualization", cv2.WINDOW_NORMAL)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        frame = tracker.find_hands(frame)
        direction = tracker.get_direction(frame)
        
        # Update the visualization
        visualizer.update_image(direction)
        
        # Display direction on frame
        cv2.putText(
            frame, 
            f"Direction: {direction}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Display the frames
        cv2.imshow("Hand Tracking", frame)
        cv2.imshow("Artistic Visualization", visualizer.current_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 