#!/usr/bin/env python3
"""
Test script to demonstrate the VibeVoice model download functionality.
This script can be used to test downloads outside of the Gradio interface.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from demo.gradio_demo import VibeVoiceDemo

def test_download():
    """Test the model download functionality."""
    
    # Create a demo instance (without loading a model)
    print("🧪 Testing VibeVoice model download functionality...")
    
    # We'll create a minimal demo instance just for testing downloads
    # Note: This will fail to load the actual model, but that's okay for testing downloads
    try:
        demo = VibeVoiceDemo(model_path="/tmp/test-path")
    except Exception as e:
        print(f"ℹ️  Expected model loading error (we're just testing downloads): {e}")
        # Create a minimal instance manually for testing
        class MinimalDemo:
            def __init__(self):
                self.is_downloading = False
                self.available_models = {
                    "VibeVoice-1.5B": {
                        "repo_id": "microsoft/VibeVoice-1.5B",
                        "size": "~2.7B parameters",
                        "description": "Compact model, good for most use cases"
                    },
                    "VibeVoice-7B-Preview": {
                        "repo_id": "WestZhang/VibeVoice-Large-pt", 
                        "size": "~9.34B parameters",
                        "description": "Larger model, potentially higher quality"
                    }
                }
        
        demo = MinimalDemo()
        
        # Add the download methods manually
        from demo.gradio_demo import VibeVoiceDemo
        demo.download_model = VibeVoiceDemo.download_model.__get__(demo, MinimalDemo)
        demo.get_model_download_status = VibeVoiceDemo.get_model_download_status.__get__(demo, MinimalDemo)
    
    # Test checking status of non-existent model
    print("\n📋 Testing status check for VibeVoice-1.5B...")
    status = demo.get_model_download_status("VibeVoice-1.5B")
    print(f"Status: {status}")
    
    # Test model info
    print("\n📝 Available models:")
    for model_name, info in demo.available_models.items():
        print(f"  - {model_name}: {info['description']} ({info['size']})")
        print(f"    Repository: {info['repo_id']}")
    
    print("\n✅ Download functionality test completed!")
    print("\n💡 To actually download a model, use the Gradio interface or call:")
    print("   for progress in demo.download_model('VibeVoice-1.5B'):")
    print("       print(progress)")

if __name__ == "__main__":
    test_download()
