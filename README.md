# Sign Language to Text Conversion

A web app that recognizes sign language gestures and converts them to text. A model is trained on a gesture dataset with `train_model.py`, and `app.py` serves the web interface that uses the trained model.

## Project structure

```
Sign-language-translator/
├── data/                 # Dataset (NOT in this repo, download from Google Drive)
├── static/               # CSS, JS, images
├── templates/            # HTML templates
├── uploads/              # Uploaded files at runtime
├── app.py                # Web app
├── train_model.py        # Model training script
├── labels.json           # Class labels
├── modelnet_model.h5     # Trained model (created by training)
├── requirements.txt      # Python dependencies
├── Dockerfile
└── docker-compose.yml
```

## Prerequisites

- [Git](https://github.com/sneha1090)
- Python 3.9, 3.10 or 3.11 (TensorFlow does not support the newest Python versions, so 3.10 is a safe choice)
- A webcam (if the app uses live camera input)

## 1. Clone the repository

```bash
git clone https://github.com/sneha1090/SignLanguage_Translator.git
cd SignLanguage_Translator
```

## 2. Download the dataset into the `data` folder

The dataset is too large for GitHub, so it is hosted on Google Drive:

**Dataset link:** `https://drive.google.com/drive/folders/1vk8NWNO7KFxIapeWrCHeCRqLvELIKfA6?usp=drive_link`

### Option A: Manual download

1. Open the Drive link above and download the folder.
2. Unzip it if needed.
3. Place the contents inside a folder named `data` in the project root, so it looks like `Sign-language-translator/data/...`.

### Option B: Download from the command line

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1vk8NWNO7KFxIapeWrCHeCRqLvELIKfA6?usp=drive_link" -O data
```

The Drive folder must be shared as **"Anyone with the link"** for `gdown` to work.

## 3. Create a virtual environment

**Windows (Command Prompt / PowerShell):**

```bash
python -m venv venv
venv\Scripts\activate
# source venv/Scripts/activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal line.

## 4. Install the requirements

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This downloads and installs every library the project needs.

## 5. Train the model

```bash
python train_model.py
```

Training can take a while depending on your dataset size and hardware. When it finishes, the trained model (`modelnet_model.h5`) and `labels.json` are ready for the app to use.

If the repository already contains a trained `modelnet_model.h5`, you can skip this step and go straight to step 6.

## 6. Run the app

```bash
python app.py
```

Then open the address printed in the terminal in your browser. For a Flask app this is usually:

```
http://127.0.0.1:5000
```

Press `Ctrl + C` in the terminal to stop the app.

## Run with Docker (optional)

If you have Docker installed, you can skip the Python setup:

```bash
docker-compose up --build
```

Make sure the `data` folder from step 2 is in place before building.

## Quick start (all commands)

**Windows:**

```bash
git clone https://github.com/Sunnykumar205/Sign-language-translator.git
cd Sign-language-translator
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install gdown
gdown --folder "PASTE_YOUR_GOOGLE_DRIVE_FOLDER_LINK_HERE" -O data
python train_model.py
python app.py
```

**macOS / Linux:**

```bash
git clone https://github.com/Sunnykumar205/Sign-language-translator.git
cd Sign-language-translator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gdown
gdown --folder "PASTE_YOUR_GOOGLE_DRIVE_FOLDER_LINK_HERE" -O data
python train_model.py
python app.py
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails on TensorFlow | Use Python 3.9 to 3.11 and make sure the venv is activated |
| `python` not recognized (Windows) | Reinstall Python and tick "Add Python to PATH" |
| `FileNotFoundError: data/...` | The dataset is missing or in the wrong place. Repeat step 2 |
| `gdown` says access denied | Set the Drive folder sharing to "Anyone with the link" |
| Port 5000 already in use | Close the other app using it or change the port in `app.py` |
| Camera not working | Allow camera permission in your browser and close other apps using the camera |
