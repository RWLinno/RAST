import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List

class LLMEncoder(nn.Module):
    """Language Model Encoder for text embeddings"""
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        """
        Args:
            model_name: Name of the pretrained model
            max_length: Maximum sequence length
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        
    def forward(self, text_list: List[str]) -> torch.Tensor:
        """
        Args:
            text_list: List of input texts
        Returns:
            Tensor of shape [batch_size, hidden_size] containing [CLS] embeddings
        """
        inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state[:, 0, :]