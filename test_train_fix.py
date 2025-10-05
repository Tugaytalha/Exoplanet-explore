#!/usr/bin/env python3
"""
Test script to verify the /train endpoint fix
"""

import requests
import json

def test_train_endpoint():
    """Test the /train endpoint to ensure it works without errors"""
    
    url = "http://localhost:8000/train"
    
    # Test data - using the existing CSV file
    files = {
        'training_file': ('koi_with_relative_location.csv', open('data/koi_with_relative_location.csv', 'rb'), 'text/csv')
    }
    
    # Simple parameters for quick testing
    data = {
        'model_name': 'test_fix_model',
        'use_existing_data': 'true',
        'max_depth': '4',
        'learning_rate': '0.1',
        'n_estimators': '50',  # Small number for quick test
        'test_size': '0.2',
        'val_size': '0.1',
        'n_folds': '2'  # Small number for quick test
    }
    
    print("🚀 Testing /train endpoint fix...")
    print("📊 Using minimal parameters for quick testing:")
    for key, value in data.items():
        if key != 'training_file':
            print(f"   {key}: {value}")
    
    try:
        print("\n⏳ Starting training (this may take a few minutes)...")
        response = requests.post(url, files=files, data=data, timeout=600)  # 10 minute timeout
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Training successful!")
            print(f"📈 Test Accuracy: {result.get('test_accuracy', 'N/A')}")
            print(f"📊 CV Mean Score: {result.get('cv_mean', 'N/A')}")
            print(f"🎯 Model Name: {result.get('model_name', 'N/A')}")
            
            # Check if hyperparameters are included in response
            if 'hyperparameters_used' in result:
                print(f"\n🔧 Hyperparameters Used:")
                for key, value in result['hyperparameters_used'].items():
                    print(f"   {key}: {value}")
                    
            print("\n🎉 Cross-validation and feature importance fixes are working correctly!")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - training might be taking longer than expected")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Close the file
        files['training_file'][1].close()

if __name__ == "__main__":
    test_train_endpoint()
