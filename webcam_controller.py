import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import threading
from collections import deque

# Disable PyAutoGUI failsafe
pyautogui.FAILSAFE = False

class WebcamController:
    def __init__(self):
        # Initialize MediaPipe Hand solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Control flags
        self.mouse_control_enabled = True
        self.click_enabled = True
        self.drag_mode = False
        self.is_clicking = False
        self.last_click_time = 0
        self.click_debounce = 0.5  # seconds

        # Click/drag/right-click state
        self.pinch_active = False
        self.pinch_started_at = 0.0
        self.drag_hold_threshold = 0.35  # seconds to hold pinch to start drag
        self.click_max_duration = 0.25   # max duration of pinch to consider a left-click

        self.right_pinch_active = False
        self.right_pinch_started_at = 0.0
        self.right_click_hold_threshold = 0.15  # require a short hold to avoid noise
        self.right_click_debounce = 0.8
        self.last_right_click_time = 0.0
        
        # Smoothing
        self.cursor_history = deque(maxlen=5)
        
        # Gesture thresholds with hysteresis (pixels)
        self.pinch_on = 35
        self.pinch_off = 45
        self.right_on = 35
        self.right_off = 45
        
        print("Webcam Controller Initialized!")
        print("Controls:")
        print("- Point with index finger to move cursor")
        print("- QUICK pinch (thumb + index) for LEFT-CLICK; HOLD pinch to DRAG")
        print("- Pinch thumb + middle finger (brief hold) for RIGHT-CLICK")
        print("- Press 'q' to quit")
        print("- Press 'm' to toggle mouse control")
        print("- Press 'c' to toggle click control")
        print("- Press SPACE to recalibrate")
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def smooth_cursor_position(self, x, y):
        """Apply smoothing to cursor movement"""
        self.cursor_history.append((x, y))
        
        if len(self.cursor_history) > 1:
            # Use weighted average for smoothing
            weights = np.linspace(0.1, 1.0, len(self.cursor_history))
            weights = weights / weights.sum()
            
            smooth_x = sum(pos[0] * weight for pos, weight in zip(self.cursor_history, weights))
            smooth_y = sum(pos[1] * weight for pos, weight in zip(self.cursor_history, weights))
            
            return int(smooth_x), int(smooth_y)
        
        return x, y
    
    def process_hand_landmarks(self, landmarks, frame_shape):
        """Process hand landmarks and return gesture information"""
        h, w, _ = frame_shape
        
        # Get landmark positions
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # Convert to pixel coordinates
        cursor_x = int(index_tip.x * self.screen_width)
        cursor_y = int(index_tip.y * self.screen_height)
        
        # Calculate distances for gestures
        thumb_index_distance = self.calculate_distance(thumb_tip, index_tip) * w
        thumb_middle_distance = self.calculate_distance(thumb_tip, middle_tip) * w
        
        # Check if index finger is extended (pointing gesture)
        index_extended = index_tip.y < index_mcp.y
        
        # Return gesture data (gesture detection now handled in handle_mouse_control)
        return {
            'cursor_pos': (cursor_x, cursor_y),
            'index_extended': index_extended,
            'thumb_index_distance': thumb_index_distance,
            'thumb_middle_distance': thumb_middle_distance
        }
    
    def handle_mouse_control(self, gesture_data):
        """Handle mouse movement and clicking based on gestures"""
        current_time = time.time()

        # Move cursor with index finger
        if self.mouse_control_enabled and gesture_data['index_extended']:
            smooth_x, smooth_y = self.smooth_cursor_position(*gesture_data['cursor_pos'])
            try:
                pyautogui.moveTo(smooth_x, smooth_y)
            except pyautogui.FailSafeException:
                pass

        # Distances
        thumb_index_distance = gesture_data['thumb_index_distance']
        thumb_middle_distance = gesture_data['thumb_middle_distance']

        # LEFT-CLICK / DRAG via thumb+index pinch with hysteresis and timing
        if self.click_enabled:
            # Update pinch active state
            if self.pinch_active:
                # Check for release
                if (thumb_index_distance > self.pinch_off) or (not gesture_data['index_extended']):
                    # Released
                    duration = current_time - self.pinch_started_at
                    if self.drag_mode:
                        try:
                            pyautogui.mouseUp()
                        except pyautogui.FailSafeException:
                            pass
                        self.drag_mode = False
                        print("Drag ended!")
                    else:
                        # Consider as a click if it was a quick pinch
                        if duration <= self.click_max_duration and (current_time - self.last_click_time) > self.click_debounce:
                            try:
                                pyautogui.click()
                                print("Left-click!")
                                self.last_click_time = current_time
                            except pyautogui.FailSafeException:
                                pass
                    self.pinch_active = False
                else:
                    # Still pinching - check if we should start a drag
                    if (not self.drag_mode) and (current_time - self.pinch_started_at >= self.drag_hold_threshold):
                        try:
                            pyautogui.mouseDown()
                        except pyautogui.FailSafeException:
                            pass
                        self.drag_mode = True
                        print("Drag started!")
            else:
                # Not active yet; check for pinch start
                if gesture_data['index_extended'] and thumb_index_distance < self.pinch_on:
                    self.pinch_active = True
                    self.pinch_started_at = current_time

        # RIGHT-CLICK via thumb+middle pinch with hysteresis and hold, debounced
        if self.click_enabled and (current_time - self.last_right_click_time > self.right_click_debounce):
            if self.right_pinch_active:
                if thumb_middle_distance > self.right_off:
                    # Released
                    duration = current_time - self.right_pinch_started_at
                    if duration >= self.right_click_hold_threshold and not self.drag_mode and not self.pinch_active:
                        try:
                            pyautogui.rightClick()
                            print("Right-click!")
                            self.last_right_click_time = current_time
                        except pyautogui.FailSafeException:
                            pass
                    self.right_pinch_active = False
            else:
                # Start right pinch only if not doing left pinch
                if (not self.pinch_active) and (thumb_middle_distance < self.right_on):
                    self.right_pinch_active = True
                    self.right_pinch_started_at = current_time
    
    def draw_info_overlay(self, frame, gesture_data):
        """Draw information overlay on the frame"""
        overlay = frame.copy()
        
        # Status indicators
        status_y = 30
        cv2.putText(overlay, f"Mouse Control: {'ON' if self.mouse_control_enabled else 'OFF'}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.mouse_control_enabled else (0, 0, 255), 2)
        
        status_y += 25
        cv2.putText(overlay, f"Click Control: {'ON' if self.click_enabled else 'OFF'}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.click_enabled else (0, 0, 255), 2)
        
        # Gesture indicators
        if gesture_data:
            status_y += 25
            pinch_color = (0, 255, 0) if self.pinch_active else (255, 255, 255)
            cv2.putText(overlay, f"Pinch idx: {gesture_data['thumb_index_distance']:.1f}px", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pinch_color, 1)
            
            status_y += 20
            right_color = (0, 255, 0) if self.right_pinch_active else (255, 255, 255)
            cv2.putText(overlay, f"Pinch mid: {gesture_data['thumb_middle_distance']:.1f}px", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
            
            status_y += 20
            drag_color = (255, 0, 0) if self.drag_mode else (255, 255, 255)
            cv2.putText(overlay, f"Drag: {'ON' if self.drag_mode else 'OFF'}", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, drag_color, 1)
        
        # Instructions
        instructions = [
            "Controls:",
            "Q - Quit",
            "M - Toggle Mouse",
            "C - Toggle Click",
            "SPACE - Recalibrate"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(overlay, instruction, (frame.shape[1] - 200, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return overlay
    
    def run(self):
        """Main loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame from webcam")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process hands
                results = self.hands.process(rgb_frame)
                gesture_data = None
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Process gestures
                        gesture_data = self.process_hand_landmarks(hand_landmarks, frame.shape)
                        
                        # Handle mouse control
                        self.handle_mouse_control(gesture_data)
                        
                        break  # Only process the first hand
                
                # Draw overlay
                frame_with_overlay = self.draw_info_overlay(frame, gesture_data)
                
                # Display frame
                cv2.imshow('Webcam Controller', frame_with_overlay)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.mouse_control_enabled = not self.mouse_control_enabled
                    print(f"Mouse control: {'enabled' if self.mouse_control_enabled else 'disabled'}")
                elif key == ord('c'):
                    self.click_enabled = not self.click_enabled
                    print(f"Click control: {'enabled' if self.click_enabled else 'disabled'}")
                elif key == ord(' '):
                    print("Recalibrating...")
                    self.cursor_history.clear()
                    if self.drag_mode:
                        pyautogui.mouseUp()
                        self.drag_mode = False
                    self.pinch_active = False
                    self.right_pinch_active = False
                    time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.drag_mode:
            try:
                pyautogui.mouseUp()
            except:
                pass
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed.")

if __name__ == "__main__":
    controller = WebcamController()
    controller.run()