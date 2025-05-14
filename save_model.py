import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def save_model(model, tokenizer, output_dir="saved_model"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the model
    model.save_pretrained(output_dir)
    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model and tokenizer saved to {output_dir}")

# Example usage:
# After training your model in the notebook:
# save_model(your_model, your_tokenizer) 