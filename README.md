# Currency Detector

## Description

This project utilizes Python and machine learning techniques to detect Indian currency denominations from images. It leverages the power of TensorFlow/Keras for building and training a robust machine learning model and OpenCV for efficient image processing.

## Motive

The primary motive behind this project is to gain a deeper understanding and practical experience in the field of machine learning. It serves as a hands-on learning opportunity to explore various concepts, algorithms, and tools involved in building a real-world image recognition application.

## Features and Working

The core functionality of this project is to accurately detect the denomination of Indian currency notes from input images. The system works by:

1.  **Image Acquisition:** Taking an image as input.
2.  **Image Preprocessing:** Using OpenCV to preprocess the image, which may include resizing, color conversion (e.g., grayscale), noise reduction, and edge detection.
3.  **Feature Extraction:** Extracting relevant features from the preprocessed image that are useful for identifying the currency denomination.
4.  **Denomination Prediction:** Feeding the extracted features into a trained machine learning model (built with TensorFlow/Keras) to predict the currency denomination.
5.  **Output:** Displaying the predicted currency denomination to the user.

## Technologies Used

*   **Python:** The primary programming language used for the entire project.
*   **Jupyter Notebook:** Used for interactive development, experimentation, and model building.
*   **Machine Learning:** The core domain of the project, encompassing model training, evaluation, and deployment.
*   **TensorFlow:** An open-source machine learning framework used for building and training the deep learning model.
*   **Keras:** A high-level API for building and training neural networks, running on top of TensorFlow.
*   **OpenCV (cv2):** A library for computer vision tasks, used for image processing and manipulation.
*   **Streamlit:** A Python library used to create and deploy interactive web applications for showcasing the currency detector.
*   **Pandas:** A library for data manipulation and analysis, used for handling datasets.
*   **NumPy:** A library for numerical computing, used for array operations and mathematical functions.
*   **Matplotlib:** A library for creating visualizations, used for plotting graphs and charts.

## Domain

This project falls under the domain of **Machine Learning**, specifically in the area of **Computer Vision** and **Image Recognition**.

## Setup Instructions

To set up and run this project, follow these steps:

1.  **Clone the Repository:**

    ```
    git clone https://github.com/your-username/currency-detector.git
    ```

2.  **Navigate to the Project Directory:**

    ```
    cd currency-detector
    ```

3.  **Install Dependencies:**

    It is highly recommended to create a virtual environment to manage project dependencies.

    ```bash
    # Create a virtual environment (optional but recommended)
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows

    # Install required packages using pip
    pip install tensorflow keras opencv-python streamlit pandas numpy matplotlib scikit-learn
    ```

4.  **Download the Dataset:**

    Download the Indian currency dataset from a reliable source (e.g., Kaggle, or create your own dataset). Place the dataset in a directory named `data` within the project directory.  The directory structure should look like this:

    ```
    currency-detector/
    ├── data/
    │   ├── denomination_10/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── denomination_20/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── ... (other denominations)
    ├── ... (other project files)
    ```

5.  **Train the Model (Optional):**

    If you want to train the model yourself (or retrain it), run the training script (e.g., `train_model.ipynb` or `train_model.py`).  Make sure to adjust the paths to the dataset in the script if necessary.

6.  **Run the Application:**

    To run the Streamlit application, use the following command:

    ```
    streamlit run app.py
    ```

    (Replace `app.py` with the actual name of your Streamlit application file).

7.  **Access the Application:**

    Open your web browser and navigate to the address displayed in the terminal (usually `http://localhost:8501`).

## Project Structure

The project structure typically includes the following files and directories:

*   `data/`: Contains the dataset of Indian currency images, organized by denomination.
*   `models/`: Stores the trained machine learning model (e.g., in `.h5` format).
*   `app.py`: The Streamlit application file for the user interface.
*   `train_model.ipynb` or `train_model.py`: The script for training the machine learning model.
*   `README.md`: This file, providing information about the project.
*   `requirements.txt`: Lists the Python packages required to run the project.
*   `utils.py` (Optional): Contains utility functions for image processing, data loading, etc.

## Future Enhancements

*   Improve the accuracy of the model by using a larger and more diverse dataset.
*   Implement data augmentation techniques to enhance the model's robustness.
*   Add support for detecting multiple currency notes in a single image.
*   Integrate the application with a mobile app for real-time currency detection.
*   Explore different machine learning models and architectures to optimize performance.
*   Add error handling and user-friendly messages to the Streamlit application.
