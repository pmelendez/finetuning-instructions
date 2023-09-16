from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
from helpers import tokenizer_factory #, print_number_of_trainable_model_parameters, print_datasets_shapes

huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)

model_name='google/flan-t5-base'

#original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
#original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenize_function = tokenizer_factory(tokenizer)


# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])
#tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)

#print_datasets_shapes(tokenized_datasets)
#print(tokenized_datasets)
#print(print_number_of_trainable_model_parameters(original_model))

index = 200

dialogue = dataset['test'][index]['dialogue']
summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

inputs = tokenizer(prompt, return_tensors='pt')
#output = tokenizer.decode(
#    original_model.generate(
#        inputs["input_ids"], 
#        max_new_tokens=200,
#    )[0], 
#    skip_special_tokens=True
#)

dash_line = '-'.join('' for x in range(100))
#print(dash_line)
#print(f'INPUT PROMPT:\n{prompt}')
#print(dash_line)
#print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
#print(dash_line)
#print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

output_dir = '/tmp' #f'./dialogue-summary-training-{str(int(time.time()))}'

lora_config = LoraConfig(
    r=8, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=1,
    max_steps=3
)

print(training_args.device)

peft_model = get_peft_model(original_model, 
                            lora_config)
#print(print_number_of_trainable_model_parameters(peft_model))

output_dir = '/tmp'#f'./peft-dialogue-summary-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=3,
    logging_steps=1,
    max_steps=3    
)
    
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)

peft_trainer.train()
peft_model_path="./peft-dialogue-summary-checkpoint-local"
peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

#trainer = Trainer(
#    model=original_model,
#    args=training_args,
#    train_dataset=tokenized_datasets['train'],
#    eval_dataset=tokenized_datasets['validation']
#)
#trainer.train()