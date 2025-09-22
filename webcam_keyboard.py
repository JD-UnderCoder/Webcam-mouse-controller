import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
from collections import deque
import threading

# Disable PyAutoGUI failsafe
pyautogui.FAILSAFE = False

class WebcamKeyboard:
    def __init__(self):
        # Initialize MediaPipe Hand solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Allow both hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Typing state
        self.typing_enabled = True
        self.current_text = ""
        self.last_gesture_time = 0
        self.gesture_debounce = 1.0  # seconds between gestures
        
        # Gesture recognition
        self.finger_count_history = deque(maxlen=5)
        self.last_finger_count = 0
        self.gesture_confirmed = False
        self.confirmation_frames = 0
        self.required_confirmation = 8  # frames
        
        # Hand position zones (divide screen into zones)
        self.zones = self.create_zones()
        self.current_zone = None
        self.zone_history = deque(maxlen=5)
        
        # Character mapping
        self.setup_character_mapping()
        
        # Mode selection
        self.current_mode = "LETTERS"  # LETTERS, NUMBERS, SYMBOLS, ACTIONS
        self.modes = ["LETTERS", "NUMBERS", "SYMBOLS", "ACTIONS"]
        self.mode_index = 0
        
        print("Webcam Keyboard Initialized!")
        print("=== CONTROLS ===")
        print("FINGER COUNT + HAND POSITION = CHARACTER")
        print("- Hold 1-5 fingers in different zones to type")
        print("- Fist (0 fingers) = SPACE")
        print("- Open palm (5 fingers) in top zone = BACKSPACE")
        print("- Two hands open = CHANGE MODE")
        print("- Press 'q' to quit")
        print("- Press 't' to toggle typing")
        print("- Press 'c' to clear text")
        print("- Press 'm' to change mode manually")
        print("==================")
        print(f"Current Mode: {self.current_mode}")
    
    def create_zones(self):
        """Create 9 zones on screen for gesture mapping"""
        return {
            'top_left': (0.0, 0.33, 0.0, 0.33),
            'top_center': (0.33, 0.66, 0.0, 0.33),
            'top_right': (0.66, 1.0, 0.0, 0.33),
            'mid_left': (0.0, 0.33, 0.33, 0.66),
            'mid_center': (0.33, 0.66, 0.33, 0.66),
            'mid_right': (0.66, 1.0, 0.33, 0.66),
            'bot_left': (0.0, 0.33, 0.66, 1.0),
            'bot_center': (0.33, 0.66, 0.66, 1.0),
            'bot_right': (0.66, 1.0, 0.66, 1.0)
        }
    
    def setup_character_mapping(self):
        """Setup character mapping for different modes"""
        self.character_maps = {
            "LETTERS": {
                1: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
                2: ['j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r'],
                3: ['s', 't', 'u', 'v', 'w', 'x', 'y', 'z', '.'],
                4: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
                5: ['J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
            },
            "NUMBERS": {
                1: ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
                2: ['0', '+', '-', '*', '/', '=', '(', ')', '%'],
                3: ['!', '@', '#', '$', '&', '?', ',', ';', ':'],
                4: ['<', '>', '[', ']', '{', '}', '|', '\\', '"'],
                5: ['~', '`', '^', '_', '\'', '.', ',', '!', '?']
            },
            "ACTIONS": {
                1: ['ENTER', 'TAB', 'ESC', 'DELETE', 'HOME', 'END', 'UP', 'DOWN', 'LEFT'],
                2: ['RIGHT', 'PGUP', 'PGDN', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
                3: ['F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'CTRL', 'ALT', 'SHIFT'],
                4: ['COPY', 'PASTE', 'CUT', 'UNDO', 'REDO', 'SAVE', 'FIND', 'REPLACE', 'SELECT'],
                5: ['CAPS', 'NUM', 'SCROLL', 'PAUSE', 'PRINT', 'INSERT', 'MENU', 'WIN', 'CMD']
            }
        }
    
    def count_fingers(self, landmarks):
        """Count extended fingers"""
        if not landmarks:
            return 0
            
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [3, 6, 10, 14, 18]
        
        fingers_up = 0
        
        # Thumb (different logic - compare x coordinates)
        if landmarks.landmark[finger_tips[0]].x > landmarks.landmark[finger_pips[0]].x:
            fingers_up += 1
        
        # Other fingers (compare y coordinates)
        for i in range(1, 5):
            if landmarks.landmark[finger_tips[i]].y < landmarks.landmark[finger_pips[i]].y:
                fingers_up += 1
                
        return fingers_up
    
    def get_hand_zone(self, landmarks):
        """Determine which zone the hand is in"""
        # Use index finger tip for zone detection
        index_tip = landmarks.landmark[8]
        x, y = index_tip.x, index_tip.y
        
        for zone_name, (x_min, x_max, y_min, y_max) in self.zones.items():
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return zone_name
        return None
    
    def get_zone_index(self, zone_name):
        """Get zone index (0-8) for character mapping"""
        zone_names = ['top_left', 'top_center', 'top_right', 
                     'mid_left', 'mid_center', 'mid_right',
                     'bot_left', 'bot_center', 'bot_right']
        try:
            return zone_names.index(zone_name)
        except ValueError:
            return 0
    
    def process_gesture(self, finger_count, zone):
        """Process gesture and return character/action"""
        current_time = time.time()
        
        if current_time - self.last_gesture_time < self.gesture_debounce:
            return None
            
        if not zone:
            return None
        
        # Special gestures
        if finger_count == 0:  # Fist = SPACE
            return ' '
        
        if finger_count == 5 and zone in ['top_left', 'top_center', 'top_right']:
            return 'BACKSPACE'
        
        # Get character from current mode
        if finger_count in self.character_maps[self.current_mode]:
            zone_index = self.get_zone_index(zone)
            char_list = self.character_maps[self.current_mode][finger_count]
            if zone_index < len(char_list):
                return char_list[zone_index]
        
        return None
    
    def execute_action(self, action):
        """Execute keyboard action"""
        try:
            if action == ' ':
                pyautogui.press('space')
                self.current_text += ' '
            elif action == 'BACKSPACE':
                pyautogui.press('backspace')
                if self.current_text:
                    self.current_text = self.current_text[:-1]
            elif action == 'ENTER':
                pyautogui.press('enter')
                self.current_text += '\\n'
            elif action == 'TAB':
                pyautogui.press('tab')
                self.current_text += '\\t'
            elif action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                pyautogui.press(action.lower())
            elif action in ['CTRL', 'ALT', 'SHIFT']:
                # These would need to be combined with other keys
                pass
            elif action == 'COPY':
                pyautogui.hotkey('ctrl', 'c')
            elif action == 'PASTE':
                pyautogui.hotkey('ctrl', 'v')
            elif action == 'CUT':
                pyautogui.hotkey('ctrl', 'x')
            elif action == 'UNDO':
                pyautogui.hotkey('ctrl', 'z')
            elif action == 'SAVE':
                pyautogui.hotkey('ctrl', 's')
            elif len(action) == 1:  # Single character
                pyautogui.typewrite(action)
                self.current_text += action
            else:
                # Other function keys
                if action.startswith('F') and action[1:].isdigit():
                    pyautogui.press(action.lower())
                    
        except Exception as e:
            print(f"Error executing action {action}: {e}")
    
    def change_mode(self):
        """Cycle through modes"""
        self.mode_index = (self.mode_index + 1) % len(self.modes)
        self.current_mode = self.modes[self.mode_index]
        print(f"Mode changed to: {self.current_mode}")
    
    def draw_interface(self, frame):
        """Draw the interface overlay"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw zones
        for zone_name, (x_min, x_max, y_min, y_max) in self.zones.items():
            x1, y1 = int(x_min * w), int(y_min * h)
            x2, y2 = int(x_max * w), int(y_max * h)
            
            # Highlight current zone
            color = (0, 255, 0) if zone_name == self.current_zone else (100, 100, 100)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Zone label
            cv2.putText(overlay, zone_name.replace('_', ' ').title(), 
                       (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Status panel
        status_y = 30
        cv2.putText(overlay, f"Mode: {self.current_mode}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        status_y += 35
        cv2.putText(overlay, f"Typing: {'ON' if self.typing_enabled else 'OFF'}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 0) if self.typing_enabled else (0, 0, 255), 2)
        
        status_y += 30
        cv2.putText(overlay, f"Fingers: {self.last_finger_count}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        status_y += 25
        cv2.putText(overlay, f"Zone: {self.current_zone or 'None'}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Current text display
        text_area_height = 100
        cv2.rectangle(overlay, (10, h - text_area_height - 10), (w - 10, h - 10), (50, 50, 50), -1)
        cv2.rectangle(overlay, (10, h - text_area_height - 10), (w - 10, h - 10), (255, 255, 255), 2)
        
        # Display current text (last 100 characters)
        display_text = self.current_text[-100:] if len(self.current_text) > 100 else self.current_text
        
        # Word wrap for display
        words = display_text.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) > 60:  # Approximate character limit per line
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
        
        # Display lines
        for i, line in enumerate(lines[-3:]):  # Show last 3 lines
            cv2.putText(overlay, line, (15, h - text_area_height + 25 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Character mapping help
        help_x = w - 400
        help_y = 30
        cv2.putText(overlay, f"{self.current_mode} Mode:", 
                   (help_x, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if self.current_mode in self.character_maps:
            for finger_count, chars in self.character_maps[self.current_mode].items():
                help_y += 25
                char_preview = ' '.join(chars[:5]) + '...' if len(chars) > 5 else ' '.join(chars)
                cv2.putText(overlay, f"{finger_count}F: {char_preview}", 
                           (help_x, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Instructions
        instructions = [
            "0 fingers = SPACE",
            "5 fingers (top) = BACKSPACE",
            "Two hands open = MODE CHANGE",
            "Q = Quit, T = Toggle, C = Clear"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(overlay, instruction, (help_x, help_y + 40 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 255), 1)
        
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
                
                total_hands = 0
                both_hands_open = False
                
                if results.multi_hand_landmarks:
                    total_hands = len(results.multi_hand_landmarks)
                    
                    # Check for mode change gesture (both hands open)
                    if total_hands == 2:
                        finger_counts = []
                        for hand_landmarks in results.multi_hand_landmarks:
                            finger_count = self.count_fingers(hand_landmarks)
                            finger_counts.append(finger_count)
                        
                        if all(count == 5 for count in finger_counts):
                            both_hands_open = True
                    
                    # Process primary hand (first detected)
                    primary_hand = results.multi_hand_landmarks[0]
                    
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, primary_hand, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Count fingers and get zone
                    finger_count = self.count_fingers(primary_hand)
                    zone = self.get_hand_zone(primary_hand)
                    
                    # Update tracking
                    self.finger_count_history.append(finger_count)
                    self.zone_history.append(zone)
                    self.current_zone = zone
                    
                    # Gesture confirmation logic
                    if len(self.finger_count_history) >= 3:
                        recent_counts = list(self.finger_count_history)[-3:]
                        recent_zones = list(self.zone_history)[-3:]
                        
                        if len(set(recent_counts)) == 1 and len(set(recent_zones)) == 1:
                            # Stable gesture
                            if not self.gesture_confirmed:
                                self.confirmation_frames += 1
                                if self.confirmation_frames >= self.required_confirmation:
                                    self.gesture_confirmed = True
                                    self.last_finger_count = finger_count
                                    
                                    # Process gesture for typing
                                    if self.typing_enabled and not both_hands_open:
                                        action = self.process_gesture(finger_count, zone)
                                        if action:
                                            self.execute_action(action)
                                            self.last_gesture_time = time.time()
                                            print(f"Action: {action}")
                        else:
                            # Reset confirmation
                            self.gesture_confirmed = False
                            self.confirmation_frames = 0
                            self.last_finger_count = finger_count
                else:
                    # No hands detected
                    self.current_zone = None
                    self.gesture_confirmed = False
                    self.confirmation_frames = 0
                
                # Handle mode change
                if both_hands_open and time.time() - self.last_gesture_time > 2.0:
                    self.change_mode()
                    self.last_gesture_time = time.time()
                
                # Draw interface
                frame_with_overlay = self.draw_interface(frame)
                
                # Display frame
                cv2.imshow('Webcam Keyboard', frame_with_overlay)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.typing_enabled = not self.typing_enabled
                    print(f"Typing: {'enabled' if self.typing_enabled else 'disabled'}")
                elif key == ord('c'):
                    self.current_text = ""
                    print("Text cleared")
                elif key == ord('m'):
                    self.change_mode()
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed.")

if __name__ == "__main__":
    keyboard = WebcamKeyboard()
    keyboard.run()