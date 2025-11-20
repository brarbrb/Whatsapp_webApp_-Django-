In this folder we have the notebook that runs basic fine tune pipeline on previously preprocessed dataset. 

`fine_tune_tiny_llama.ipynb` file creates folders that needed for loading the fine-tuned transformer later on. 

We uploaded only the final folder `bbt-lora/merged` with final weights after training. 

`evaluation.py` is used for runing our evaluation metrics. 

in folder `fine_tune_data` you can find the all jsons used for training, and more importantly the folder `output_file` that stores all `evaluation.py` results. 
