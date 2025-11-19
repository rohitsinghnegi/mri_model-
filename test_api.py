import requests
import numpy as np
from PIL import Image
import io

# Create a dummy image (red square)
img = Image.new('RGB', (224, 224), color = 'red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_byte_arr = img_byte_arr.getvalue()

url = 'http://127.0.0.1:5000/api/predict'
files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
data = {'model_type': 'mri'} # Change to 'skin' or 'xray' to test others

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    try:
        print("Response JSON:", response.json())
    except:
        print("Response Text:", response.text)

except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the server. Make sure 'python app.py' is running in another terminal.")
