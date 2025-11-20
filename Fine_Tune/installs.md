Since we had older GPU there were a lot of incompabilities in the process with newer packages. 
We installed these versions. 
```bash
pip install "numpy==1.26.4" --force-reinstall
pip uninstall -y torch torchvision torchaudio
pip install "torch==2.1.0+cu118" \
            "torchvision==0.16.0+cu118" \
            "torchaudio==2.1.0+cu118" \
            --index-url https://download.pytorch.org/whl/cu118
pip install --force-reinstall \
    "transformers==4.36.2" \
    "sentence-transformers==2.2.2" \
    "peft==0.10.0" \
    "accelerate==0.27.2" \
    "datasets==2.19.0" \
    "bitsandbytes==0.42.0"
```