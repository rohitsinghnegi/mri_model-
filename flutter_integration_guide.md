# Flutter Integration Guide

This guide explains how to connect your Flutter app to the deployed Python Flask API.

## 1. Add Dependencies
Add `http` and `image_picker` to your `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  http: ^1.2.0
  image_picker: ^1.0.7
```

Run `flutter pub get`.

## 2. Create API Service
Create a file `lib/services/api_service.dart`:

```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  // REPLACE with your actual Render/Heroku URL after deployment
  // For local testing on emulator use 'http://10.0.2.2:5000/api/predict'
  static const String baseUrl = "https://your-app-name.onrender.com/api/predict";

  Future<Map<String, dynamic>> predictImage(File imageFile, String modelType) async {
    var request = http.MultipartRequest('POST', Uri.parse(baseUrl));
    
    // Add model type field
    request.fields['model_type'] = modelType; // 'mri', 'skin', or 'xray'

    // Add image file
    request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));

    try {
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception("Failed to predict: ${response.body}");
      }
    } catch (e) {
      throw Exception("Error connecting to API: $e");
    }
  }
}
```

## 3. Usage in UI
Here is a simple example of how to use it in your widget:

```dart
// Inside your State class
File? _image;
String _result = "";
final ApiService _apiService = ApiService();

Future<void> _pickAndPredict() async {
  final picker = ImagePicker();
  final pickedFile = await picker.pickImage(source: ImageSource.gallery);

  if (pickedFile != null) {
    setState(() {
      _image = File(pickedFile.path);
      _result = "Analyzing...";
    });

    try {
      // Change 'mri' to 'skin' or 'xray' as needed
      final response = await _apiService.predictImage(_image!, 'mri');
      
      setState(() {
        _result = "Prediction: ${response['predicted_class']}\n"
                  "Confidence: ${response['confidence'].toStringAsFixed(2)}%";
      });
    } catch (e) {
      setState(() {
        _result = "Error: $e";
      });
    }
  }
}
```

## 4. Deployment to Render (Free Tier)
1.  Push your code to **GitHub**.
2.  Go to [Render.com](https://render.com) and create a **New Web Service**.
3.  Connect your GitHub repository.
4.  **Build Command**: `pip install -r requirements.txt`
5.  **Start Command**: `gunicorn app:app`
6.  Click **Create Web Service**.
7.  Copy the URL provided by Render and paste it into `api_service.dart`.
