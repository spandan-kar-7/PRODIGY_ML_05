# Food Image Recognition and Calorie Estimation

A beginner-friendly machine learning project that classifies food images into five categories using a TensorFlow/Keras CNN and displays an approximate calorie reference.

## Food categories
- Pizza
- Hamburger
- Sushi
- Ice cream
- French fries

## Features
- Upload a food image in a simple Streamlit interface
- Predict one of five food categories
- Display the model's predicted class probability
- Show a predefined calorie reference for a typical serving

> **Important limitations:** This project classifies only five categories. The calorie values are rough reference estimates, not nutrition measurements from the image. Actual calories depend on serving size, recipe, ingredients, and preparation. Model confidence is not a guarantee of correctness.

## Project structure

```text
food-recognition-calorie-estimation/
├── app.py
├── food_classes.json
├── requirements.txt
├── .gitignore
├── README.md
├── food_recognition_model.keras   # Add your trained model; not included in this starter pack
└── notebooks/
    └── Task_05_Food_Recognition.ipynb  # Export your Colab notebook here
```

## Setup

### 1. Install Python
Use Python 3.10 or 3.11 if possible.

### 2. Download your trained model
Save your `food_recognition_model.keras` file in the project root. Do not commit a large model file to GitHub unless you have checked repository size limits; use Git LFS or a release if needed.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open in your browser. Upload a JPG, JPEG, or PNG food image.

## Export your Colab notebook
In Google Colab, select **File → Download → Download .ipynb**. Save the downloaded notebook in the `notebooks/` folder as `Task_05_Food_Recognition.ipynb`.

## Upload to GitHub
1. Sign in at https://github.com/
2. Create a new repository named `food-recognition-calorie-estimation`.
3. Choose Public only if you are comfortable sharing the project. Do not upload credentials, private data, or anything you do not want public.
4. Upload the project files and your notebook.
5. Add the trained model separately if it is too large for a normal GitHub upload.

## Suggested resume description
**Food Image Recognition and Calorie Estimation | Python, TensorFlow/Keras, CNN, Streamlit**
- Built a CNN-based image classifier for five food categories using the Food-101 dataset.
- Evaluated the model using validation metrics, a classification report, a confusion matrix, and training curves.
- Created a simple image-upload interface that displays predicted category, confidence, and approximate reference calories.

Replace this description with your actual measured results before publishing. Do not claim an accuracy percentage unless it comes from your experiment.
