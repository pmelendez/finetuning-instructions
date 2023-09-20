from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftConfig, PeftModel
from helpers import get_model_output, get_prompt_dialog_from_dataset, get_tokenized_datasets, print_number_of_trainable_model_parameters, print_output_comparison, tokenizer_factory 
huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)
#model_name='google/flan-t5-base'
model_name='./peft-dialogue-summary-checkpoint-local'

#original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
#original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenized_datasets = get_tokenized_datasets(tokenizer, dataset)

prompt = get_prompt_dialog_from_dataset(dataset, 200)
output = get_model_output(tokenizer, original_model, prompt)

lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

peft_model = get_peft_model(original_model, 
                            lora_config)

output_dir = './peft-dialogue-summary-training'#f'./peft-dialogue-summary-training-{str(int(time.time()))}'
peft_model_path="./peft-dialogue-summary-checkpoint-local"

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=1,
    logging_steps=1,
    save_strategy="steps"
    #max_steps=24   
)
    
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)

try:
    peft_trainer.train()
except KeyboardInterrupt:
    print("Interrupting but still saving")


peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)


#----
mymodelname='./peft-dialogue-summary-checkpoint-local'
mymodel = AutoModelForSeq2SeqLM.from_pretrained(mymodelname)
output_new = get_model_output(tokenizer, mymodel, prompt)
print_output_comparison(prompt, output, output_new)
