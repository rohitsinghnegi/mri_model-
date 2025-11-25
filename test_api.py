import requests
import numpy as np
from PIL import Image
import io

def test_model(model_type, color='red'):
    """Test a specific model with a dummy image"""
    # Create a dummy image
    img = Image.new('RGB', (224, 224), color=color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    url = 'http://127.0.0.1:5000/api/predict'
    files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
    data = {'model_type': model_type}
    
    try:
        print(f"\n{'='*60}")
        print(f"Testing {model_type.upper()} model with {color} image...")
        print(f"{'='*60}")
        
        response = requests.post(url, files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS!")
            print(f"   Model Used: {result.get('model_used')}")
            print(f"   Predicted Class: {result.get('predicted_class')}")
            print(f"   Confidence: {result.get('confidence'):.2f}%")
        else:
            print(f"\n❌ ERROR!")
            try:
                print(f"   Response: {response.json()}")
            except:
                print(f"   Response Text: {response.text}")
        
        return response.status_code == 200
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the server.")
        print("   Make sure 'python app.py' is running in another terminal.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("MEDICAL IMAGE CLASSIFIER - API TEST SUITE")
    print("="*60)
    print("\nThis script tests all three models:")
    print("1. MRI (Brain Tumor)")
    print("2. Skin (Skin Disease)")
    print("3. X-Ray (Lung Disease)")
    print("\nEach model will be tested with a dummy colored image.")
    print("="*60)
    
    # Test all three models
    models = ['mri', 'skin', 'xray']
    colors = ['red', 'blue', 'green']  # Different colors for variety
    
    results = {}
    for model, color in zip(models, colors):
        results[model] = test_model(model, color)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for model, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{model.upper():10s}: {status}")
    print("="*60)
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! All models are responding correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    print()

if __name__ == "__main__":
    main()
