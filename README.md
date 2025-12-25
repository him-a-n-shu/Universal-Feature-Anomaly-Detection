# Universal Feature Anomaly Detection (UFAD)

This project is a full-stack application for industrial anomaly detection in images. It uses a two-stage approach:

1.  **Universal Feature Extractor:** A `wide_resnet50_2` backbone (the `universal_feature_extractor.pth`) is pre-trained on a large collection of industrial images to learn general-purpose features.
2.  **On-Site Adaptation:** For a new product, the factory engineer uploads a small set of "golden sample" (good) images. The system performs on-site adaptation by building a product-specific "coreset" of features.
3.  **Live Inference:** New test images are compared against this coreset to find anomalies, generating a decision ("Normal" or "Anomalous"), an anomaly score, and a visual heatmap.

This system is designed to be highly adaptable to new product lines without requiring a full model retrain for each one.

## Core Technology

* **Backend:** Python, Flask, PyTorch, scikit-learn (NearestNeighbors)
* **Frontend:** React, Vite, Tailwind CSS
* **Model:** `wide_resnet50_2`

## Datasets Used

The universal feature extractor was trained on images from the following public datasets:

* **MVTec-AD:** [https://www.kaggle.com/datasets/ipythonx/mvtec-ad](https://www.kaggle.com/datasets/ipythonx/mvtec-ad)
* **VisA:** [https://www.kaggle.com/datasets/ess1004/visa-anomaly-detection](https://www.kaggle.com/datasets/ess1004/visa-anomaly-detection)

---

## Project Structure

```
/your-project-directory
├── app.py                      # The Python Flask backend server
├── requirements.txt            # Python dependencies for the backend
├── universal_feature_extractor.pth # The pre-trained model file
├── README.md                   # This setup guide
│
└── frontend/                   # The React project folder
    ├── index.html              # Main HTML file (loads Tailwind)
    ├── package.json            # Frontend dependencies
    └── src/
        ├── App.jsx             # The main React application component
        ├── index.css           # Tailwind CSS imports
        └── main.jsx            # React entry point
```
---

## Setup & Running Instructions

### Prerequisites

* **Python 3.8+** and `pip`
* **Node.js 18+** and `npm`
* **Your Model File:** You MUST have your `universal_feature_extractor.pth` file in the main project directory.

### Part 1: Run the Backend (Python Server)

1.  **Open your first terminal** in this main project directory.

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    * On macOS/Linux: `source venv/bin/activate`
    * On Windows: `venv\Scripts\activate`

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask Server:**
    ```bash
    python app.py
    ```
    The server will start on `http://localhost:5001`. **Leave this terminal running.**

### Part 2: Run the Frontend (React App)

1.  **Open a NEW (second) terminal.** Navigate to the main project directory.

2.  **Navigate into the `frontend` folder:**
    *(If you haven't created it yet, follow the "Creating the Frontend" steps below)*
    ```bash
    cd frontend
    ```

3.  **Install Node Modules (if you haven't yet):**
    ```bash
    npm install
    ```

4.  **Start the React App:**
    ```bash
    npm run dev
    ```
    Your browser should automatically open to `http://localhost:5173` (or a similar port).

---

## How to Use the Application

1.  **Step 1: Product Setup**
    * Navigate to the **Product Setup** tab.
    * Enter a **Product Name** (e.g., "bottle", "pcb", "carpet").
    * Drag and drop (or click to upload) several "golden sample" (good) images of that product.
    * Click **"Generate Product Coreset"**. The backend will process these images and save a new `{product_name}_coreset.pkl` file.
    * You will see the new product appear in the "Product Status" list.

2.  **Step 2: Live Inference**
    * Navigate to the **Live Inference** tab.
    * Select your newly created product from the **"Select Product"** dropdown.
    * Upload a single **Test Image** (this can be a good or bad product).
    * Click **"Run Detection"**.
    * The system will display the **Decision** (Normal/Anomalous), the **Anomaly Score**, the original image, and a heatmap highlighting any detected anomalies.

---

## Team Members

| Name | Student ID |
| :--- | :--- |
| D Barghav | 2023UME0253 |
| HIMANSHU  | 2023UCS0092 |
| Purushartha Gupta | 2023UCE0062 |
| Nitin Kumar Yadav | 2023UCS0104 |
| Aman Nagar | 2023UME0242 |
