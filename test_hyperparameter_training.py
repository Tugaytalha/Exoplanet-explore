#!/usr/bin/env python3
"""
Test script for hyperparameter training endpoint
"""

import requests
import json

def test_hyperparameter_training():
    """Test the /train endpoint with custom hyperparameters"""
    
    url = "http://localhost:8000/train"
    
    # Test data - using the existing CSV file
    files = {
        'training_file': ('koi_with_relative_location.csv', open('data/koi_with_relative_location.csv', 'rb'), 'text/csv')
    }
    
    # Custom hyperparameters
    data = {
        'model_name': 'test_hyperparameter_model',
        'use_existing_data': 'true',
        'max_depth': '8',
        'learning_rate': '0.1',
        'n_estimators': '200',
        'min_child_weight': '5',
        'gamma': '0.2',
        'subsample': '0.9',
        'colsample_bytree': '0.9',
        'reg_alpha': '0.2',
        'reg_lambda': '0.2',
        'early_stopping_rounds': '30',
        'test_size': '0.2',
        'val_size': '0.1',
        'n_folds': '3'
    }
    
    print("🚀 Testing hyperparameter training endpoint...")
    print(f"📊 Hyperparameters:")
    for key, value in data.items():
        if key != 'training_file':
            print(f"   {key}: {value}")
    
    try:
        response = requests.post(url, files=files, data=data, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Training successful!")
            print(f"📈 Test Accuracy: {result.get('test_accuracy', 'N/A')}")
            print(f"📊 CV Mean Score: {result.get('cv_mean', 'N/A')}")
            print(f"🎯 Model Name: {result.get('model_name', 'N/A')}")
            print(f"📁 Download Link: {result.get('download_link', 'N/A')}")
            
            # Check if hyperparameters are included in response
            if 'hyperparameters_used' in result:
                print(f"\n🔧 Hyperparameters Used:")
                for key, value in result['hyperparameters_used'].items():
                    print(f"   {key}: {value}")
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
    test_hyperparameter_training()
