# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HF_TOKEN = "hf_"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'meta-llama/Prompt-Guard-86M',
            cache_dir="checkpoints",
            token=HF_TOKEN
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Prompt-Guard-86M',
            cache_dir="checkpoints",
            token=HF_TOKEN
        )

    def predict(
        self,
        prompt: str = Input(description="Input text"),
    ) -> str:
        """Run a single prediction on the model"""
        texts = [prompt]
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding='longest',
            truncation=True,
            max_length=512
        )

        # pass through the model and get probabilities
        logits = self.model(**batch).logits
        probs = torch.softmax(logits, dim=-1)

        # Format output
        results = []
        for i, text in enumerate(texts):
            predicted_class = probs[i].argmax().item()
            if predicted_class in self.model.config.id2label:
                label = self.model.config.id2label[predicted_class]
            else:
                label = f"UNKNOWN_CLASS_{predicted_class}"
            
            result = {
                "labels": [label],
                "scores": [probs[i].max().item()]
            }
            results.append(result)

        json_result = {
            "results": results,
        }

        result = json.dumps(json_result, indent=2)

        return result