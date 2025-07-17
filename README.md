# American Sign Language Dataset Preprocessing with Real-Time Prediction

This repository contains a Python-based project for collecting images of American Sign Language (ASL) gestures, extracting hand landmarks using MediaPipe, and training a machine learning model (XGBoost) for real-time character prediction.

## Features
- Captures ASL gesture images from webcam (A-Z)
- Extracts hand landmarks using MediaPipe Hands module
- Calculates distances and angles from landmarks
- Converts data into a CSV format for machine learning pipelines
- Trains an XGBoost model with hyperparameter tuning
- Performs real-time ASL character recognition using a webcam
- Automatically constructs a sentence from recognized characters

## Prerequisites
Before running the project, ensure you have the following installed:

- Python 3.7 or higher
- OpenCV
- MediaPipe 0.10.5
- NumPy
- Pandas
- Scikit-learn
- XGBoost

You can install the required libraries with:
```bash
pip install opencv-python mediapipe xgboost numpy pandas scikit-learn
```

## Folder Structure
Organize your dataset folder (`./data`) as follows after collecting gesture images:
```
./data
    /A
        1.jpg
        2.jpg
        ...
    /B
        1.jpg
        2.jpg
        ...
    ...
```
Each subfolder corresponds to an ASL letter and contains images of hands making that gesture.

## Running the Code

1. **Collect Images for ASL Dataset**
   Run this script to capture 200 images per class (A–Z) from your webcam:
   ```bash
   python collect_images.py
   ```

2. **Generate CSV from Collected Images**
   Extracts hand keypoints, distances, and angles to create a structured dataset:
   ```bash
   python create_dataset_csv.py
   ```

3. **Train the XGBoost Model**
   Trains a multi-class classifier using the generated CSV and saves the model:
   ```bash
   python train_model.py
   ```

4. **Run Real-Time ASL Prediction**
   Uses the webcam and trained model to recognize ASL letters and build a sentence:
   ```bash
   python realtime_inference.py
   ```

## Debugging Tips
- If the script prints "No hands detected" for multiple images, ensure that:
  - Hands are fully visible and centered in the camera frame
  - Lighting is adequate and background is clear
- Ensure each class folder contains readable `.jpg` images
- The generated feature vector must be of consistent length (81 values)

## Contributing
If you'd like to improve this project, feel free to fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
