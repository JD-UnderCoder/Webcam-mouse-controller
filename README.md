# Webcam Mouse Control Project

Control your computer's mouse using hand gestures captured through your webcam! This project uses computer vision and machine learning to detect hand landmarks and translate gestures into mouse actions.

## Features

### 🖱️ Mouse Control (`webcam_controller.py`)
- **Cursor Movement**: Point with your index finger to move the cursor
- **Left Click**: Quick pinch (thumb + index finger) 
- **Right Click**: Pinch thumb + middle finger with brief hold
- **Drag & Drop**: Hold pinch gesture to start dragging
- **Real-time Hand Tracking**: Uses MediaPipe for accurate hand detection
- **Smooth Cursor Movement**: Built-in smoothing algorithm for natural movement
- **Toggle Controls**: Enable/disable mouse and click controls on-the-fly

### ⌨️ Virtual Keyboard (`webcam_virtual_keyboard.py`)
- **Full QWERTY Layout**: Complete virtual keyboard with all standard keys
- **Dwell Selection**: Hover over keys to select them
- **Pinch Selection**: Alternative pinch-to-click input method
- **Modifier Keys**: Support for Shift, Caps Lock, Ctrl, Alt
- **Real-time Preview**: See what you're typing on screen

### 🎯 Gesture Keyboard (`webcam_keyboard.py`)
- **Zone-based Typing**: Divide screen into 9 zones for character input
- **Multiple Modes**: Letters, Numbers, Symbols, and Actions
- **Finger Counting**: Use different finger counts (1-5) for character selection
- **Hand Position Mapping**: Combine finger count with hand position for precise input

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- PyAutoGUI (`pyautogui`)
- NumPy (`numpy`)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JD-UnderCoder/Webcam-mouse-controller.git
   cd Webcam-mouse-controller
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the mouse controller:**
   ```bash
   python webcam_controller.py
   ```

## Usage

### Mouse Control Mode

1. **Run the main controller:**
   ```bash
   python webcam_controller.py
   ```

2. **Hand Gestures:**
   - **Move Cursor**: Point with your index finger
   - **Left Click**: Quick pinch (thumb + index finger together, then release quickly)
   - **Right Click**: Pinch thumb + middle finger, hold briefly, then release
   - **Drag**: Pinch thumb + index finger and hold for 0.35+ seconds to start dragging

3. **Keyboard Controls:**
   - `q` - Quit the application
   - `m` - Toggle mouse control on/off
   - `c` - Toggle click control on/off  
   - `SPACE` - Recalibrate (reset all gestures)

### Virtual Keyboard Mode

1. **Run the virtual keyboard:**
   ```bash
   python webcam_virtual_keyboard.py
   ```

2. **Selection Methods:**
   - **Dwell**: Hover your fingertip over a key for 0.7 seconds
   - **Pinch**: Point at a key and pinch to select it

3. **Controls:**
   - `p` - Toggle between dwell and pinch selection
   - `q` - Quit

### Gesture Keyboard Mode

1. **Run the gesture keyboard:**
   ```bash
   python webcam_keyboard.py
   ```

2. **How to Type:**
   - Position your hand in one of 9 screen zones
   - Show 1-5 fingers to select character sets
   - Hold the gesture steady to confirm selection

3. **Special Gestures:**
   - **Fist (0 fingers)**: Space
   - **Open hand (5 fingers) in top zone**: Backspace
   - **Both hands open**: Change mode (Letters/Numbers/Actions)

## Configuration

### Gesture Sensitivity
You can adjust gesture detection thresholds in the code:

```python
# In webcam_controller.py
self.pinch_on = 35        # Pinch detection threshold (pixels)
self.pinch_off = 45       # Pinch release threshold (pixels)
self.drag_hold_threshold = 0.35  # Time to hold for drag (seconds)
```

### Camera Settings
Modify camera resolution and settings:

```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

## Technical Details

- **Hand Detection**: Uses Google's MediaPipe for real-time hand landmark detection
- **Gesture Recognition**: Custom algorithms to detect pinch gestures and finger positions
- **Smoothing**: Implements weighted moving average for smooth cursor movement
- **Hysteresis**: Prevents gesture flickering with on/off thresholds
- **Debouncing**: Prevents accidental multiple clicks

## Troubleshooting

### Camera Issues
- Ensure your webcam is connected and working
- Try changing the camera index: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`
- Check if other applications are using the camera

### Performance Issues
- Reduce camera resolution for better performance
- Ensure good lighting for better hand detection
- Keep hands clearly visible in the camera frame

### Gesture Detection Problems
- Adjust gesture thresholds if gestures are too sensitive or not sensitive enough
- Ensure clear background behind your hands
- Try recalibrating with the SPACE key

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) by Google for hand tracking
- [OpenCV](https://opencv.org/) for computer vision functionality
- [PyAutoGUI](https://pyautogui.readthedocs.io/) for system automation

## Future Enhancements

- [ ] Voice commands integration
- [ ] Gesture customization interface
- [ ] Multi-hand gesture support
- [ ] Eye tracking integration
- [ ] Mobile app companion
- [ ] Gesture recording and playback
- [ ] Machine learning model for custom gestures

---

**Note**: This software is designed for accessibility and convenience. Please use responsibly and ensure you have appropriate permissions when using automation features.
