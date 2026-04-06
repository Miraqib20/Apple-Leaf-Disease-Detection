# 🍎 Apple Orchard Doctor System

A web-based disease and pest identification system for apple growers in Kashmir Valley.
Now includes a **CNN-based leaf scanner** powered by MobileNetV2 + TensorFlow.

---

## Project Structure

```
project/
├── app.py                        # Flask backend — all routes
├── requirements.txt              # Python dependencies
├── database/
│   └── schema.sql                # MySQL schema + seed data (9 diseases)
├── cnn/
│   ├── train.py                  # CNN training script (MobileNetV2)
│   ├── predictor.py              # Inference module used by Flask
│   ├── __init__.py
│   ├── apple_disease_model.h5    # generated after training
│   ├── class_names.json          # generated after training
│   ├── training_plot.png         # generated after training
│   └── confusion_matrix.png      # generated after training
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── identify.html             # Symptom-based identification
│   ├── scan.html                 # CNN leaf photo scanner
│   ├── disease_detail.html
│   ├── diseases.html
│   ├── advice.html
│   └── new_advice.html
└── static/
    ├── css/style.css
    ├── js/main.js
    └── uploads/                  # Uploaded leaf images stored here
```

---

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up MySQL database
```bash
mysql -u root -p < database/schema.sql
```

### 3. Configure database connection
Edit `app.py` or set environment variables:
```bash
export DB_HOST=localhost
export DB_USER=root
export DB_PASSWORD=your_password
export DB_NAME=orchard_doctor
export SECRET_KEY=your-secret-key
```

### 4. Run the web app
```bash
python app.py
```
Open: **http://localhost:5000**

---

## CNN Setup (Leaf Disease Scanner)

The CNN scan feature requires training before it works.

### Step 1 — Download the dataset
Go to: https://www.kaggle.com/datasets/emmarex/plantdisease

Download and extract. Keep only the 4 Apple folders:
```
data/
  Apple___Apple_scab/
  Apple___Black_rot/
  Apple___Cedar_apple_rust/
  Apple___healthy/
```

### Step 2 — Train the model
```bash
python cnn/train.py --data_dir data/ --epochs 15
```

Training runs in two phases automatically:
- **Phase 1** (15 epochs): Trains the classification head. MobileNetV2 base is frozen.
- **Phase 2** (10 epochs): Fine-tunes the top 30 layers of the base for higher accuracy.

Expected accuracy: **93–97%** on the PlantVillage apple subset.

Output files saved to `cnn/`:
| File | Description |
|------|-------------|
| `apple_disease_model.h5` | Trained Keras model |
| `class_names.json` | Label index mapping |
| `training_plot.png` | Accuracy/loss curves |
| `confusion_matrix.png` | Per-class performance |

### Step 3 — Use the scanner
Go to **Scan Leaf** in the navbar. Upload a leaf photo and the system will:
1. Pre-process the image (resize to 224×224, normalise)
2. Run it through the CNN
3. Return predicted disease + confidence score for all 4 classes
4. Look up the disease in MySQL
5. Display all recommended treatments
6. Auto-log the diagnosis to your account

---

## CNN Architecture

```
Input (224 × 224 × 3)
       ↓
MobileNetV2 (ImageNet pretrained)
  Phase 1: fully frozen
  Phase 2: top 30 layers unfrozen
       ↓
GlobalAveragePooling2D
       ↓
BatchNormalization
       ↓
Dense(256, ReLU) → Dropout(0.4)
       ↓
Dense(128, ReLU) → Dropout(0.3)
       ↓
Dense(4, Softmax)
  Apple Scab | Black Rot | Cedar Apple Rust | Healthy
```

MobileNetV2 was chosen because it runs on CPU (no GPU required),
has a small model file (~14 MB), and achieves high accuracy on plant disease tasks.

---

## Disease Classes

| CNN Label | Database Disease | Severity |
|-----------|-----------------|----------|
| Apple Scab | Apple Scab | High |
| Black Rot | Black Rot | High |
| Cedar Apple Rust | Cedar Apple Rust | Medium |
| Healthy | *(no treatment needed)* | — |

Additional diseases (Fire Blight, Powdery Mildew, Codling Moth, etc.)
are identifiable through the symptom checker at `/identify`.

---

## All Routes

| Route | Feature | Access |
|-------|---------|--------|
| `/` | Landing page | Public |
| `/register` | Create account | Public |
| `/login` | Log in | Public |
| `/dashboard` | Stats + quick links | Logged in |
| `/scan` | CNN leaf photo scanner | Logged in |
| `/identify` | Symptom-based checker | Logged in |
| `/diseases` | Full disease library | Logged in |
| `/disease/<id>` | Disease + treatments | Logged in |
| `/advice` | Expert advice posts | Logged in |
| `/advice/new` | Post advice | Expert / Manager |

---

## Tech Stack

- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Backend:** Python / Flask
- **Database:** MySQL
- **CNN:** TensorFlow / Keras — MobileNetV2 transfer learning
- **Fonts:** Playfair Display + DM Sans (Google Fonts)
