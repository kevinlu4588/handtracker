import cv2
import mediapipe as mp
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
import threading
import queue
import time

class ImageGenerator:
    def __init__(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        if torch.backends.mps.is_available():
            self.pipeline = self.pipeline.to("mps")
        self.image_queue = queue.Queue(maxsize=1)
        self.current_direction = None
        self.running = True
        self.thread = threading.Thread(target=self._generate_images)
        self.thread.start()

    def _generate_images(self):
        while self.running:
            if self.current_direction is not None:
                prompt = self._get_prompt_for_direction(self.current_direction)
                try:
                    image = self.pipeline(
                        prompt,
                        num_inference_steps=20,
                        guidance_scale=7.5
                    ).images[0]
                    
                    # Convert PIL image to OpenCV format
                    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Resize image to a reasonable size
                    image = cv2.resize(image, (512, 512))
                    
                    # Update queue with new image
                    if not self.image_queue.full():
                        self.image_queue.put(image)
                except Exception as e:
                    print(f"Error generating image: {e}")
                time.sleep(0.1)  # Prevent excessive CPU usage

    def _get_prompt_for_direction(self, direction):
        base_prompt = "digital art of a hand pointing {}, minimalist style, clean lines, high contrast"
        direction_map = {
            "left": "to the left",
            "right": "to the right",
            "up": "upward",
            "down": "downward"
        }
        return base_prompt.format(direction_map.get(direction, "forward"))

    def update_direction(self, direction):
        if direction != self.current_direction:
            self.current_direction = direction

    def get_current_image(self):
        try:
            return self.image_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.thread.join()

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def get_direction(self):
        if not self.results.multi_hand_landmarks:
            return None

        hand_landmarks = self.results.multi_hand_landmarks[0]
        
        # Get index finger tip and middle finger tip
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        
        # Calculate direction based on finger position
        dx = index_tip.x - middle_tip.x
        dy = index_tip.y - middle_tip.y
        
        # Determine direction based on the larger difference
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"

def main():
    cap = cv2.VideoCapture(0)
    hand_tracker = HandTracker()
    image_generator = ImageGenerator()
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Find hands and get direction
            frame = hand_tracker.find_hands(frame)
            direction = hand_tracker.get_direction()
            
            # Update image generator with new direction
            if direction:
                image_generator.update_direction(direction)
            
            # Get current generated image
            generated_image = image_generator.get_current_image()
            
            # Display the frames
            cv2.imshow('Hand Tracking', frame)
            if generated_image is not None:
                cv2.imshow('Generated Image', generated_image)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        image_generator.stop()

if __name__ == "__main__":
    main() 