# Ring Try-On Project

## 📌 Overview
This project is designed for **virtual ring try-on** using **computer vision** and **hand tracking**. It utilizes **MediaPipe** for real-time hand landmark detection and integrates with Blender for rendering.

## 📂 Project Structure
```
ringTryOn/
│── blender/                      # Blender integration for rendering
│   ├── blender_renderer.py       # Blender script for rendering rings
│
│── general_research/             # Research & experiments
│   ├── research.ipynb            # Jupyter Notebook with experiments
│
│── mediapipe/                     # Hand tracking implementation
│   ├── hand_tracking_image.py     # Processes static images
│   ├── hand_tracking_realtime.py  # Real-time hand tracking with a webcam
│
│── ring_try_on_input_data/        # Folder for storing input images, videos & depth maps
│
│── venv/                          # Virtual environment (should be ignored in Git)
│
│── .gitignore                     # Git ignore file (make sure venv/ is ignored)
│── main.py                         # Main execution script
│── requirements.txt                # Python dependencies
```

## 🛠 Installation
### 1️⃣ Clone the repository
```sh
git clone <repository_url>
cd ringTryOn
```

### 2️⃣ Create a virtual environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies
```sh
pip install -r requirements.txt
```

## 🚀 Usage
### 1️⃣ **Run Hand Tracking on Images**
```sh
python mediapipe/hand_tracking_image.py
```

### 2️⃣ **Run Real-Time Hand Tracking**
```sh
python mediapipe/hand_tracking_realtime.py
```

## 📝 Notes
- Ensure that **`.gitignore`** includes `venv/` to avoid pushing the virtual environment.
- The **input data** (hand images, depth maps) should be placed in `ring_try_on_input_data/`.
- **MediaPipe** is required for hand tracking.
- **Blender** is used for rendering ring placement.

## 📌 Future Enhancements
- Add **pipeline** for automation of process.
- Improve **ring placement accuracy**.
- Implement **customizable ring designs**.
- Add **LiDAR-based depth mapping** for precise placement.

## 🤝 Contributing
Pull requests are welcome! If you want to improve this project, feel free to fork and submit PRs.

