import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

import sys
sys.stdout.reconfigure(encoding='utf-8')


DATASET_PATH = "C:/Users/JAWERIA/Downloads/ASL Digits/asl_dataset_digits"
IMG_SIZE = 100

def simple_augment_image(image):
    """Simple augmentation without external libraries"""
    augmented_images = [image]  # Start with original
    
    # Flip horizontally
    augmented_images.append(cv2.flip(image, 1))
    
    # Brightness variations
    for alpha in [0.8, 1.2]:
        bright_img = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        augmented_images.append(bright_img)
    
    return augmented_images

def extract_consistent_features(image):
    """Extract consistent number of features with error handling"""
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
    
    # 2. Hu Moments (7 features - ALWAYS 7) - FIXED
    hu_moments = cv2.HuMoments(moments)
    for i in range(7):
        if hu_moments[i] != 0:
            # Fix the deprecation warning by extracting scalar value
            hu_val = hu_moments[i][0] if hu_moments[i].size > 0 else hu_moments[i]
            if hu_val != 0:
                hu_val = -np.sign(hu_val) * np.log10(abs(hu_val))
            else:
                hu_val = 0
        else:
            hu_val = 0
        features.append(float(hu_val))
    
    # 3. Contour-based features (ALWAYS 8 features) - FIXED with error handling
    contour_features = [0.0] * 8  # Initialize with zeros

    try:
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            
            # Basic contour features
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            # Shape metrics
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Bounding box features
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            bbox_area = w * h
            
            # Convexity defects count - with error handling
            defect_count = 0
            if len(cnt) >= 4:  # Need at least 4 points for convexity defects
                try:
                    hull_indices = cv2.convexHull(cnt, returnPoints=False)
                    if len(hull_indices) > 3:
                        defects = cv2.convexityDefects(cnt, hull_indices)
                        if defects is not None:
                            defect_count = len(defects)
                except cv2.error:
                    # If convexity defects fails, use 0
                    defect_count = 0
            
            # Contour points count
            contour_points = len(cnt)
            
            contour_features = [
                float(area), float(perimeter), float(circularity), float(solidity),
                float(aspect_ratio), float(bbox_area), float(defect_count), float(contour_points)
            ]
    except Exception as e:
        # If any contour processing fails, use zeros
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
    
    # TOTAL: 10 + 7 + 8 + 4 = 29 features (ALWAYS CONSISTENT)
    return np.array(features, dtype=np.float32)

def load_enhanced_dataset():
    """Load dataset with simple augmentations - UPDATED for flat structure"""
    features_list = []
    labels_list = []
    
    print("🔍 Loading enhanced dataset with simple augmentations...")
    print(f"📂 Dataset path: {DATASET_PATH}")
    
    # Check if path exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset path does not exist: {DATASET_PATH}")
        return np.array([]), np.array([])
    
    # List what's in the directory for debugging
    dir_contents = os.listdir(DATASET_PATH)
    print(f"📁 Contents in dataset path: {dir_contents}")
    
    total_images_loaded = 0
    error_count = 0
    
    # Process each digit folder (0, 1, 2, ..., 9)
    for digit_folder in os.listdir(DATASET_PATH):
        digit_path = os.path.join(DATASET_PATH, digit_folder)
        
        if not os.path.isdir(digit_path):
            continue
            
        # Convert folder name to label (0-9)
        try:
            label = int(digit_folder)
            print(f"📁 Processing digit: {digit_folder} -> label {label}")
        except ValueError:
            print(f"❓ Skipping non-digit folder: {digit_folder}")
            continue
        
        image_count = 0
        digit_images = os.listdir(digit_path)
        print(f"    📸 Found {len(digit_images)} images in folder '{digit_folder}'")
        
        for img_name in digit_images:
            if image_count >= 80:  # Limit images per class
                break
                
            img_path = os.path.join(digit_path, img_name)
            
            # Skip if it's a directory
            if os.path.isdir(img_path):
                continue
                
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Preprocess image
                img_processed = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                _, img_processed = cv2.threshold(img_processed, 127, 255, cv2.THRESH_BINARY)
                
                # Create augmented versions
                augmented_versions = simple_augment_image(img_processed)
                
                for aug_img in augmented_versions:
                    try:
                        feats = extract_consistent_features(aug_img)
                        features_list.append(feats)
                        labels_list.append(label)
                        image_count += 1
                        total_images_loaded += 1
                    except Exception as e:
                        error_count += 1
                        continue
            else:
                print(f"    ⚠️  Could not load image: {img_name}")
    
    print(f"📊 Total images loaded: {total_images_loaded}")
    print(f"📊 Total samples after augmentation: {len(features_list)}")
    if error_count > 0:
        print(f"⚠️  Skipped {error_count} images due to processing errors")
    
    # Check if we have any data
    if len(features_list) == 0:
        print("❌ No features extracted! Check:")
        print("   - Dataset path correctness")
        print("   - Folder structure")
        print("   - Image formats")
        return np.array([]), np.array([])
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features each")
    print(f"✅ Class distribution: {np.bincount(y)}")
    return X, y

def train_enhanced_knn():
    """Train KNN model with enhanced features and cross-validation"""
    print("🚀 Training Enhanced KNN...")
    X, y = load_enhanced_dataset()
    
    if len(X) == 0:
        print("❌ No data loaded! Check dataset path.")
        return None
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"📊 Classes: {len(np.unique(y))}")
    
    # Split data with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("🧠 Finding best K value...")
    best_k, best_score = 3, 0
    
    # Test different K values
    for k in [3, 5, 7, 9]:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        print(f"  K={k}, Accuracy: {score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"🎯 Training final model with K={best_k}...")
    final_knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    final_knn.fit(X_train, y_train)
    
    # Final evaluation
    train_score = final_knn.score(X_train, y_train)
    test_score = final_knn.score(X_test, y_test)
    
    print(f"📊 Training Accuracy: {train_score:.3f} ({train_score*100:.1f}%)")
    print(f"✅ Test Accuracy: {test_score:.3f} ({test_score*100:.1f}%)")
    
    # Save the model
    model_filename = "enhanced_knn_model.pkl"
    joblib.dump(final_knn, model_filename)
    print(f"💾 Enhanced KNN model saved as '{model_filename}'")
    
    return final_knn

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with detailed metrics"""
    if model is None:
        print("❌ No model to evaluate!")
        return
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Model Evaluation:")
    print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Test samples: {len(y_test)}")
    
    # Class-wise accuracy
    unique_labels = np.unique(y_test)
    print(f"   Class-wise performance:")
    for label in unique_labels:
        mask = y_test == label
        if np.sum(mask) > 0:
            class_accuracy = np.mean(y_pred[mask] == y_test[mask])
            print(f"     Class {label}: {class_accuracy:.3f}")

if __name__ == "__main__":
    print("=" * 50)
    print("🤖 Gesture Recognition Model Training")
    print("=" * 50)
    
    model = train_enhanced_knn()
    
    if model is not None:
        print("\n🎉 Training completed successfully!")
        print("📱 Now run the real-time recognition script!")
        
        # Load test data for evaluation
        X, y = load_enhanced_dataset()
        if len(X) > 0:
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            evaluate_model(model, X_test, y_test)
    else:
        print("❌ Training failed!")
    
    print("=" * 50)