#!/usr/bin/env python3
"""Check GPU availability and status for ABBA embedding generation."""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.embeddings import EmbeddingModelManager


def check_gpu_status():
    """Display comprehensive GPU status information."""
    print("=== ABBA GPU Status Check ===\n")
    
    # Basic CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    
    if torch.cuda.is_available():
        # GPU details
        print(f"\nGPU Information:")
        print(f"  Device count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i}:")
            print(f"    Name: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            
            # Current memory usage
            if i == torch.cuda.current_device():
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"    Memory allocated: {allocated:.2f} GB")
                print(f"    Memory reserved: {reserved:.2f} GB")
        
        # Current device
        print(f"\nCurrent device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Test tensor creation
        print("\nTesting GPU tensor creation...")
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            print("✓ Successfully created tensor on GPU")
            del test_tensor
        except Exception as e:
            print(f"✗ Failed to create tensor on GPU: {e}")
    else:
        print("\n⚠️  No GPU detected - embeddings will be generated on CPU")
        print("This will be significantly slower than GPU processing.")
        
        # Check why CUDA might not be available
        print("\nPossible reasons:")
        print("  1. No NVIDIA GPU installed")
        print("  2. NVIDIA drivers not installed")
        print("  3. PyTorch installed without CUDA support")
        print("  4. CUDA version mismatch")
        
        print("\nTo install PyTorch with CUDA support:")
        print("  Visit: https://pytorch.org/get-started/locally/")
    
    # Test with EmbeddingModelManager
    print("\n=== Testing ABBA Embedding Model Manager ===")
    try:
        manager = EmbeddingModelManager()
        print(f"Model manager initialized with device: {manager.device}")
        
        # Get model info
        english_info = manager.get_model_info("english")
        multilingual_info = manager.get_model_info("multilingual")
        
        print(f"\nEnglish model ({english_info['name']}):")
        print(f"  Dimensions: {english_info['dimensions']}")
        print(f"  Max length: {english_info['max_length']}")
        
        print(f"\nMultilingual model ({multilingual_info['name']}):")
        print(f"  Dimensions: {multilingual_info['dimensions']}")
        print(f"  Max length: {multilingual_info['max_length']}")
        
    except Exception as e:
        print(f"Error initializing model manager: {e}")
    
    # Performance estimates
    print("\n=== Performance Estimates ===")
    if torch.cuda.is_available():
        print("With GPU acceleration:")
        print("  - Verse embeddings: ~100-500 verses/second")
        print("  - Word embeddings: ~500-2000 words/second")
        print("  - Full Bible (~31k verses): ~1-5 minutes")
    else:
        print("With CPU only:")
        print("  - Verse embeddings: ~10-50 verses/second")
        print("  - Word embeddings: ~50-200 words/second")
        print("  - Full Bible (~31k verses): ~10-50 minutes")
    
    print("\n✓ GPU status check complete!")


if __name__ == "__main__":
    check_gpu_status()