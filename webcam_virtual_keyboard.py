import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
from collections import deque

pyautogui.FAILSAFE = False

class VirtualKeyboard:
    def __init__(self):
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Keyboard layout (approx real widths in units)
        self.layout = [
            # Row 1
            [("`",1), ("1",1), ("2",1), ("3",1), ("4",1), ("5",1), ("6",1), ("7",1), ("8",1), ("9",1), ("0",1), ("-",1), ("=",1), ("Backspace",2)],
            # Row 2
            [("Tab",1.5), ("Q",1), ("W",1), ("E",1), ("R",1), ("T",1), ("Y",1), ("U",1), ("I",1), ("O",1), ("P",1), ("[",1), ("]",1), ("\\",1.5)],
            # Row 3
            [("Caps",1.75), ("A",1), ("S",1), ("D",1), ("F",1), ("G",1), ("H",1), ("J",1), ("K",1), ("L",1), (";",1), ("'",1), ("Enter",2)],
            # Row 4
            [("Shift",2.25), ("Z",1), ("X",1), ("C",1), ("V",1), ("B",1), ("N",1), ("M",1), (",",1), (".",1), ("/",1), ("Shift",2.25)],
            # Row 5
            [("Ctrl",1.25), ("Win",1.25), ("Alt",1.25), ("Space",6), ("Alt",1.25), ("Menu",1.25), ("Ctrl",1.25)]
        ]

        # Key rectangles computed each frame
        self.key_rects = []  # list of dicts: {label, x1,y1,x2,y2}

        # Selection method: 'dwell' or 'pinch'
        self.selection_method = 'dwell'
        self.dwell_threshold = 0.7  # seconds to select on hover
        self.hover_key = None
        self.hover_started_at = 0.0
        self.select_debounce = 0.35
        self.last_select_time = 0.0

        # Pinch detection (thumb-index)
        self.pinch_active = False
        self.pinch_on = 35  # px
        self.pinch_off = 45 # px
        self.pinch_started_at = 0.0

        # Modifiers
        self.caps_lock = False
        self.shift_latched = False  # applies to next character

        # Preview buffer for on-screen display
        self.preview_text = ""
        self.preview_max = 60

        print("Virtual Keyboard Initialized")
        print("Controls:")
        print("- Hover over a key to select (dwell) or pinch to click")
        print("- Press 'p' to toggle selection method (dwell/pinch)")
        print("- Press 'q' to quit")

    def shifted_char(self, ch):
        shift_map = {
            '`':'~','1':'!','2':'@','3':'#','4':'$','5':'%','6':'^','7':'&','8':'*','9':'(','0':')',
            '-':'_','=':'+','[':'{',']':'}','\\':'|',';':':','\'':'"',',':'<','.':'>','/':'?'
        }
        return shift_map.get(ch, ch.upper() if ch.isalpha() else ch)

    def type_key(self, label):
        now = time.time()
        if now - self.last_select_time < self.select_debounce:
            return
        self.last_select_time = now

        try:
            if label == 'Backspace':
                pyautogui.press('backspace')
                if self.preview_text:
                    self.preview_text = self.preview_text[:-1]
                return
            if label == 'Tab':
                pyautogui.press('tab'); self.preview_text += '\t'; return
            if label == 'Enter':
                pyautogui.press('enter'); self.preview_text += '\n'; return
            if label == 'Space':
                pyautogui.press('space'); self.preview_text += ' '; return
            if label == 'Caps':
                self.caps_lock = not self.caps_lock; return
            if label == 'Shift':
                self.shift_latched = True; return
            if label in ['Ctrl','Alt','Win','Menu']:
                # Not implemented as modifiers here to avoid sticky states
                return

            # Character keys
            out = label
            if len(out) == 1:
                if out.isalpha():
                    upper = self.caps_lock ^ self.shift_latched
                    out = out.upper() if upper else out.lower()
                else:
                    if self.shift_latched:
                        out = self.shifted_char(out)

                pyautogui.typewrite(out)
                self.preview_text += out
                self.preview_text = self.preview_text[-self.preview_max:]
            else:
                # Non-char labels already handled above
                pass
        finally:
            # Shift applies once
            if self.shift_latched and label not in ['Shift','Caps']:
                self.shift_latched = False

    def calc_keyboard_rects(self, frame):
        h, w = frame.shape[:2]
        margin = 10
        top = int(h * 0.55)
        bottom = h - margin
        kb_height = bottom - top
        row_gap = 6
        available_h = kb_height - row_gap * (len(self.layout) + 1)
        row_h = int(available_h / len(self.layout))

        rects = []
        y = top + row_gap
        for row in self.layout:
            # total units in row
            total_units = sum(k[1] for k in row)
            col_gap = 6
            available_w = w - 2*margin - col_gap * (len(row) + 1)
            unit_w = available_w / total_units
            x = margin + col_gap
            for label, units in row:
                key_w = int(units * unit_w)
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + key_w), int(y + row_h)
                rects.append({
                    'label': label,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
                x += key_w + col_gap
            y += row_h + row_gap
        self.key_rects = rects

    def draw_keyboard(self, frame, hover_label=None, dwell_progress=0.0):
        overlay = frame.copy()
        for r in self.key_rects:
            x1,y1,x2,y2 = r['x1'],r['y1'],r['x2'],r['y2']
            label = r['label']
            # coloring
            base_color = (60,60,60)
            if label in ['Caps'] and self.caps_lock:
                base_color = (80,120,60)
            if label in ['Shift'] and self.shift_latched:
                base_color = (80,80,140)
            cv2.rectangle(overlay, (x1,y1), (x2,y2), base_color, -1)
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (200,200,200), 2)
            # label text
            text = label
            font_scale = 0.6 if len(label) <= 3 else 0.5
            (tw,th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx = x1 + (x2-x1 - tw)//2
            ty = y1 + (y2-y1 + th)//2
            cv2.putText(overlay, text, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1)
            # hover highlight and dwell bar
            if hover_label == label:
                cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
                if self.selection_method == 'dwell' and dwell_progress > 0:
                    bar_w = int((x2-x1) * min(1.0, dwell_progress))
                    cv2.rectangle(overlay, (x1,y2-6), (x1+bar_w,y2-2), (0,255,0), -1)
        return overlay

    def fingertip_and_pinch(self, landmarks, frame):
        h, w = frame.shape[:2]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_mcp = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        # pixel position
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)
        # distances in pixels (normalize by width)
        dist = np.hypot((index_tip.x - thumb_tip.x)*w, (index_tip.y - thumb_tip.y)*w)
        index_extended = index_tip.y < index_mcp.y
        return (x,y), dist, index_extended

    def key_at(self, x, y):
        for r in self.key_rects:
            if r['x1'] <= x <= r['x2'] and r['y1'] <= y <= r['y2']:
                return r['label']
        return None

    def run(self):
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("Failed to read from camera")
                    break
                frame = cv2.flip(frame, 1)

                # Compute keyboard rects for current frame
                self.calc_keyboard_rects(frame)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)

                hover_label = None
                dwell_progress = 0.0

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    self.mp_drawing.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

                    (fx,fy), pinch_dist, index_ext = self.fingertip_and_pinch(hand, frame)
                    cv2.circle(frame, (fx,fy), 8, (0,255,255), -1)

                    # Determine hovered key by fingertip
                    hover_label = self.key_at(fx, fy)

                    now = time.time()
                    # Dwell selection
                    if self.selection_method == 'dwell' and hover_label:
                        if hover_label != self.hover_key:
                            self.hover_key = hover_label
                            self.hover_started_at = now
                        dwell_progress = (now - self.hover_started_at) / self.dwell_threshold
                        if (now - self.hover_started_at) >= self.dwell_threshold:
                            self.type_key(hover_label)
                            self.hover_started_at = now  # allow multiple presses with dwell
                    else:
                        # reset dwell tracking when not in dwell or not hovering
                        self.hover_key = hover_label
                        self.hover_started_at = now

                    # Pinch selection (on release while pointing)
                    if self.selection_method == 'pinch' and index_ext and hover_label:
                        if self.pinch_active:
                            if pinch_dist > self.pinch_off:
                                # release -> select
                                self.type_key(hover_label)
                                self.pinch_active = False
                        else:
                            if pinch_dist < self.pinch_on:
                                self.pinch_active = True

                # Draw keyboard overlay
                frame_out = self.draw_keyboard(frame, hover_label, dwell_progress)

                # Status bar
                status = f"Method: {self.selection_method.upper()}  Caps: {'ON' if self.caps_lock else 'OFF'}  Shift: {'ON' if self.shift_latched else 'OFF'}"
                cv2.putText(frame_out, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                # Preview text
                if self.preview_text:
                    cv2.putText(frame_out, self.preview_text[-60:], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)

                cv2.imshow('Virtual Keyboard', frame_out)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.selection_method = 'pinch' if self.selection_method == 'dwell' else 'dwell'
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Cleanup completed.")

if __name__ == '__main__':
    vk = VirtualKeyboard()
    vk.run()
