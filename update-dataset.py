import os
from dotenv import load_dotenv
load_dotenv()

print("Updating dataset...")

# Thiết lập biến môi trường cho Kaggle API
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

# Tạo file cấu hình cho Kaggle nếu chưa có
kaggle_dir = os.path.expanduser('~/.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)
with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:
    import json
    json.dump({
        'username': os.environ['KAGGLE_USERNAME'],
        'key': os.environ['KAGGLE_KEY']
    }, f)
os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)

import kaggle

# Tải dataset (tự động giải nén)
kaggle.api.dataset_download_files(
    'nelgiriyewithana/global-weather-repository',  # Thay bằng dataset bạn muốn
    path='./data',
    unzip=True
)

print("Finish!")