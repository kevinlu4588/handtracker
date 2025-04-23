import cv2
import mediapipe as mp
import numpy as np

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
        
        # Get coordinates of key points
        wrist = hand_landmarks.landmark[0]
        index_finger_tip = hand_landmarks.landmark[8]
        middle_finger_tip = hand_landmarks.landmark[12]
        
        # Calculate vectors
        index_vector = np.array([
            index_finger_tip.x - wrist.x,
            index_finger_tip.y - wrist.y
        ])
        
        # Normalize vector
        magnitude = np.sqrt(index_vector[0]**2 + index_vector[1]**2)
        if magnitude == 0:
            return "No clear direction"
            
        index_vector = index_vector / magnitude
        
        # Determine direction based on vector components
        if abs(index_vector[0]) > abs(index_vector[1]):
            # Horizontal movement
            if index_vector[0] > 0:
                return "Right"
            else:
                return "Left"
        else:
            # Vertical movement
            if index_vector[1] > 0:
                return "Down"
            else:
                return "Up"

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Find hands and get direction
        frame = tracker.find_hands(frame)
        direction = tracker.get_direction(frame)
        
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
        
        # Display the frame
        cv2.imshow("Hand Tracking", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 