#!/usr/bin/env python3
"""
Test script for AutoML pipeline
"""

import sys
import os

def test_automl():
    """Test the AutoML pipeline with minimal configuration."""
    print("🧪 Testing AutoML Pipeline...")
    
    # Import the AutoML pipeline
    try:
        from train_automl import AutoMLPipeline, CONFIG
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Create a minimal configuration for testing
    test_config = CONFIG.copy()
    test_config.update({
        'n_trials': 5,  # Very small number for quick testing
        'timeout_per_model': 60,  # 1 minute timeout
        'output_dir': 'test_automl_outputs'
    })
    
    print(f"📊 Test configuration:")
    print(f"   - Trials per model: {test_config['n_trials']}")
    print(f"   - Timeout: {test_config['timeout_per_model']}s")
    print(f"   - Output directory: {test_config['output_dir']}")
    
    try:
        # Initialize AutoML pipeline
        automl = AutoMLPipeline(test_config)
        
        # Test data loading and preprocessing
        print("\n🔍 Testing data loading and preprocessing...")
        X, y = automl.load_and_preprocess_data()
        
        print(f"✅ Data loaded successfully:")
        print(f"   - Features: {X.shape[1]}")
        print(f"   - Samples: {X.shape[0]}")
        print(f"   - Target classes: {y.unique()}")
        
        # Test model configurations
        print("\n🔍 Testing model configurations...")
        model_configs = automl.get_model_configurations()
        print(f"✅ Found {len(model_configs)} model types:")
        for model_name in model_configs.keys():
            print(f"   - {model_name}")
        
        print("\n🎉 AutoML pipeline test completed successfully!")
        print("💡 To run full AutoML, use: python train_automl.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_automl()
    sys.exit(0 if success else 1)
