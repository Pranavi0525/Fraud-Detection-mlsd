import os

base_path = r"D:\SaiU\semester-6\Porjects\Fraud_Detection"

structure = {
    "docker-compose.yml": "",
    
    "transaction_service": {
        "main.py": "",
        "requirements.txt": ""
    },
    
    "fraud_detection_service": {
        "main.py": "",
        "rules.py": "",
        "ml_scorer.py": "",
        "requirements.txt": ""
    },
    
    "alert_service": {
        "main.py": "",
        "requirements.txt": ""
    },
    
    "models": {
        "champion_model.joblib": "",
        "scaler.joblib": "",
        "feature_names.json": "",
        "champion_metadata.json": ""
    },
    
    "README.md": ""
}

def create_structure(base, struct):
    for name, content in struct.items():
        path = os.path.join(base, name)
        
        if "." in name:  # File
            with open(path, "w") as f:
                f.write(content)
        else:  # Folder
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)

# Create base folder
os.makedirs(base_path, exist_ok=True)

# Create everything
create_structure(base_path, structure)

print("✅ Project structure created successfully!")