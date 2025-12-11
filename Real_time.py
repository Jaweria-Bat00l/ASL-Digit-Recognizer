import cv2
import numpy as np
import joblib
from collections import deque
import sys
sys.stdout.reconfigure(encoding='utf-8')


# Constants (must match training script)
IMG_SIZE = 100
MODEL_PATH = "enhanced_knn_model.pkl"

def extract_consistent_features(image):
    """Extract consistent number of features with error handling - EXACT COPY FROM TRAINING"""
    features = []
    
    # 1. Basic moments (10 features)
    moments = cv2.moments(image)
    moment_features = [
        moments['m00'] if moments['m00'] != 0 else 0,
        moments['m10'] if moments['m10'] != 0 else 0,
        moments['m01'] if moments['m01'] != 0 else 0,
        moments['mu20'] if moments['mu20'] != 0 else 0,
        moments['mu11'] if moments['mu11'] != 0 else 0,
        moments['mu02'] if moments['mu02'] != 0 else 0,
        moments['mu30'] if moments['mu30'] != 0 else 0,
        moments['mu21'] if moments['mu21'] != 0 else 0,
        moments['mu12'] if moments['mu12'] != 0 else 0,
        moments['mu03'] if moments['mu03'] != 0 else 0
    ]
    features.extend(moment_features)
    
    # 2. Hu Moments (7 features - ALWAYS 7)
    hu_moments = cv2.HuMoments(moments)
    for i in range(7):
        if hu_moments[i] != 0:
            hu_val = hu_moments[i][0] if hu_moments[i].size > 0 else hu_moments[i]
            if hu_val != 0:
                hu_val = -np.sign(hu_val) * np.log10(abs(hu_val))
            else:
                hu_val = 0
        else:
            hu_val = 0
        features.append(float(hu_val))
    
    # 3. Contour-based features (ALWAYS 8 features)
    contour_features = [0.0] * 8

    try:
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            solidity = area / hull_area if hull_area > 0 else 0
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            bbox_area = w * h
            
            defect_count = 0
            if len(cnt) >= 4:
                try:
                    hull_indices = cv2.convexHull(cnt, returnPoints=False)
                    if len(hull_indices) > 3:
                        defects = cv2.convexityDefects(cnt, hull_indices)
                        if defects is not None:
                            defect_count = len(defects)
                except cv2.error:
                    defect_count = 0
            
            contour_points = len(cnt)
            
            contour_features = [
                float(area), float(perimeter), float(circularity), float(solidity),
                float(aspect_ratio), float(bbox_area), float(defect_count), float(contour_points)
            ]
    except Exception as e:
        contour_features = [0.0] * 8

    features.extend(contour_features)
    
    # 4. Image statistics (4 features - ALWAYS 4)
    stats_features = [
        float(np.mean(image)),
        float(np.std(image)),
        float(np.median(image)),
        float(np.max(image))
    ]
    features.extend(stats_features)
    
    return np.array(features, dtype=np.float32)

def preprocess_hand_roi(roi):
    """Preprocess hand ROI to match training data format"""
    # Convert to grayscale if needed
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    # Resize to match training size
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    
    # Apply binary threshold (same as training)
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    
    return binary

def setup_roi(frame):
    """Setup region of interest in the frame"""
    height, width = frame.shape[:2]
    
    # Define ROI rectangle (center of frame)
    roi_size = min(300, width - 100, height - 100)
    x1 = (width - roi_size) // 2
    y1 = (height - roi_size) // 2 - 50  # Move up slightly for better hand placement
    x2 = x1 + roi_size
    y2 = y1 + roi_size
    
    return (x1, y1, x2, y2)

def apply_skin_mask(frame, roi_coords):
    """Apply skin color masking to improve hand detection"""
    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        return roi
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create skin mask
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to original ROI
    masked_roi = cv2.bitwise_and(roi, roi, mask=skin_mask)
    
    return masked_roi

def real_time_gesture_recognition():
    """Real-time gesture recognition using webcam"""
    
    # Load the trained model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        print(f"📊 Model type: {type(model).__name__}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🤖 Please run the training script first to generate enhanced_knn_model.pkl")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("📹 Webcam initialized successfully")
    print("🎮 Controls:")
    print("   - Press 'q' to quit")
    print("   - Press 'c' to clear prediction history")
    print("   - Press 's' to toggle skin mask")
    print("   - Place your hand in the green rectangle")
    
    # Prediction smoothing
    prediction_history = deque(maxlen=7)
    current_prediction = None
    confidence = 0
    use_skin_mask = True
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        
        # Get ROI coordinates
        x1, y1, x2, y2 = setup_roi(frame)
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Place hand here", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Extract and process ROI
        if use_skin_mask:
            processed_roi = apply_skin_mask(frame, (x1, y1, x2, y2))
            roi_display = processed_roi.copy()
        else:
            processed_roi = frame[y1:y2, x1:x2]
            roi_display = processed_roi.copy()
        
        # Only process every 3rd frame to reduce CPU load
        frame_count += 1
        should_process = (frame_count % 3 == 0) and processed_roi.size > 0
        
        if should_process:
            try:
                # Convert to grayscale for feature extraction
                if len(processed_roi.shape) == 3:
                    gray_roi = cv2.cvtColor(processed_roi, cv2.COLOR_BGR2GRAY)
                else:
                    gray_roi = processed_roi
                
                # Preprocess ROI (resize and threshold)
                preprocessed_roi = preprocess_hand_roi(gray_roi)
                
                # Extract features (same as training)
                features = extract_consistent_features(preprocessed_roi)
                
                # Reshape for prediction
                features = features.reshape(1, -1)
                
                # Make prediction
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                
                # Get confidence
                confidence = np.max(probabilities) * 100
                
                # Add to history for smoothing
                prediction_history.append(prediction)
                
                # Get most frequent prediction from history
                if prediction_history:
                    current_prediction = max(set(prediction_history), 
                                           key=prediction_history.count)
                
                # Display processed ROI for debugging (top-left corner)
                debug_roi = cv2.resize(preprocessed_roi, (150, 150))
                debug_roi_bgr = cv2.cvtColor(debug_roi, cv2.COLOR_GRAY2BGR)
                frame[10:160, 10:160] = debug_roi_bgr
                
                # Display feature count (for verification)
                cv2.putText(frame, f"Features: {len(features[0])}", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            except Exception as e:
                # print(f"⚠️ Prediction error: {e}")  # Uncomment for debugging
                pass
        
        # Display prediction and confidence
        prediction_text = "No hand detected"
        confidence_text = ""
        color = (0, 0, 255)  # Red for no detection
        
        if current_prediction is not None and confidence > 50:  # Increased confidence threshold
            prediction_text = f"Digit: {current_prediction}"
            confidence_text = f"Confidence: {confidence:.1f}%"
            
            # Change color based on confidence
            if confidence > 80:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 65:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 165, 255)  # Orange - low confidence
            
            # Draw prediction with background for better visibility
            text_size = cv2.getTextSize(prediction_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            cv2.rectangle(frame, (x1, y2 + 20), (x1 + text_size[0] + 10, y2 + 90), (0, 0, 0), -1)
            
        # Always display the prediction text
        cv2.putText(frame, prediction_text, (x1 + 5, y2 + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        if confidence_text:
            cv2.putText(frame, confidence_text, (x1 + 5, y2 + 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display skin mask status
        mask_status = "Skin Mask: ON" if use_skin_mask else "Skin Mask: OFF"
        cv2.putText(frame, mask_status, (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display instructions
        cv2.putText(frame, "Press 'q': Quit | 'c': Clear | 's': Toggle Mask", (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('ASL Digits - Real-time Gesture Recognition', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            prediction_history.clear()
            current_prediction = None
            confidence = 0
            print("🗑️ Prediction history cleared")
        elif key == ord('s'):
            use_skin_mask = not use_skin_mask
            mask_status = "ON" if use_skin_mask else "OFF"
            print(f"🎭 Skin mask: {mask_status}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Real-time recognition ended")

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 ASL Digits - Real-time Gesture Recognition")
    print("=" * 60)
    
    real_time_gesture_recognition()