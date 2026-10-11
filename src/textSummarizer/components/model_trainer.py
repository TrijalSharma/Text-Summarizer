import os
import torch
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from textSummarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # 1. Hardware Check: Route the math to your RTX 4060 if available, else CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 2. Load the Google Pegasus Tokenizer and Model into your GPU
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        
        # 3. Data Collator: This grabs your dataset and dynamically groups it into batches
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        # 4. Load the mathematical dataset we generated in the previous stage
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # 5. Inject the hyperparameters we defined in params.yaml
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=float(self.config.save_steps),
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )

        # 6. Initialize the Hugging Face Trainer engine
        trainer = Trainer(
            model=model_pegasus, 
            args=trainer_args,
            processing_class=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"], 
            eval_dataset=dataset_samsum_pt["validation"]
        )
        
        # 7. Pull the trigger (this is where your GPU fans will spin up)
        trainer.train()

        # 8. Save the newly smart model and its dictionary back to your hard drive
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))