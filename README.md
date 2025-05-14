```
███████████                    █████      █████████    █████████   █████████ 
░░███░░░░░░█                   ░░███      ███░░░░░███  ███░░░░░███ ███░░░░░███
 ░███   █ ░   ██████    █████  ███████   ███     ░░░  ███     ░░░ ░███    ░░░ 
 ░███████    ░░░░░███  ███░░  ░░░███░   ░███         ░███         ░░█████████ 
 ░███░░░█     ███████ ░░█████   ░███    ░███         ░███          ░░░░░░░░███
 ░███  ░     ███░░███  ░░░░███  ░███ ███░░███     ███░░███     ███ ███    ░███
 █████      ░░████████ ██████   ░░█████  ░░█████████  ░░█████████ ░░█████████ 
░░░░░        ░░░░░░░░ ░░░░░░     ░░░░░    ░░░░░░░░░    ░░░░░░░░░   ░░░░░░░░░
             A Molecular Collision Cross Section Predictor | by Amir Aghajan
```
#  FastCCS

> **FastCCS** is a deep learning-based web platform for predicting Molecular Collision Cross Section (CCS) values from molecular SMILES and adduct ions.  
> It leverages state-of-the-art models to offer real-time predictions via a simple web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Django](https://img.shields.io/badge/Built%20With-Django-red?logo=django)

---


## 🧪 Features

- Predict CCS from SMILES + adduct ion.
- Real-time prediction via web interface.
- Integrated with trained models 
- Modular and extensible design.

---

## ⚙️ Installation & Setup

## ⚠️ Prerequisites
- **Python 3**: Required to run the application.

### 1. Clone the repository
You can get the project files in one of two ways: by downloading the ZIP file or by cloning the repository using Git.

### Option 1: Download the ZIP File

- Click on the **"Code"** button.
- Select **"Download ZIP"**.
- Once the ZIP file is downloaded, extract it to your desired location on your computer.

### Option 2: Clone the Repository Using Git

If you prefer to clone the repository directly to your system, you can use Git. Follow these steps:

- Open a terminal (Command Prompt, PowerShell, or any terminal of your choice).
- Run the following commands:

   ```bash
   git clone https://github.com/AmirAg47/FastCCS.git
   cd FastCCS

### 2. (Recommended) Create virtual environment
```bash
python -m venv venv
🔹 On Windows: venv\Scripts\activate
🔹 On macOS/Linux: source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run database migrations
```bash
python manage.py migrate
```

### 5. Run the development server
```bash
python manage.py runserver
```

Then open your browser and visit:

```
http://127.0.0.1:8000/predict/
```



## 📁 Project Structure

```
FastCCS/
│
├── ccspredictor/         # Main Django app
├── models/               # Trained model
├── predictor/            # Prediction logic
├── static/               # Static CSS/JS assets
├── media/                # Files and result CSV
├── manage.py             # Django entry point
├── requirements.txt      # Python dependencies
└── README.md             # You're here!
```

---

## 📝 License

This project is licensed under the **MIT License** – see the [LICENSE](./LICENSE) file for details.

---

## ☕ Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## ✉️ Contact

Feel free to reach out via GitHub or raise an issue.

Happy predicting! ⚗️✨
