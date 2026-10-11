from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import evaluate
import pandas as pd
import torch
from tqdm import tqdm
from textSummarizer.entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Split the dataset into smaller batches that we can process simultaneously."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer, batch_size=16, device="cuda"):
        article_batches = list(self.generate_batch_sized_chunks(dataset['dialogue'], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset['summary'], batch_size))

        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
            inputs = tokenizer(article_batch, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
            
            # Generate summaries using the fine-tuned model
            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device), 
                length_penalty=0.8, 
                num_beams=8, 
                max_length=128
            )
            
            # Decode the generated text
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
            
            # Send the predictions and actual answers to the Rouge metric calculator
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
            
        # Compute the final mathematical overlap scores
        score = metric.compute()
        return score

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
        
        # Load dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        
        # Load Rouge metric using the Hugging Face evaluate library
        rouge_metric = evaluate.load("rouge")
        
        # Calculate metric on the test dataset (Using batch_size=2 to protect VRAM during generation)
        score = self.calculate_metric_on_test_ds(
            dataset_samsum_pt['test'], rouge_metric, model, tokenizer, batch_size=2, device=device
        )
        
        # Create a pandas dataframe and save it to CSV
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = dict((rn, score[rn]) for rn in rouge_names)
        
        df = pd.DataFrame(rouge_dict, index=['pegasus'])
        df.to_csv(self.config.metric_file_name, index=False)