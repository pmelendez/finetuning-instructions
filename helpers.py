from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer

def tokenizer_factory(tokenizer):
    def tokenize_function(example):
        start_prompt = 'Summarize the following conversation.\n\n'
        end_prompt = '\n\nSummary: '
        prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
        example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
        example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
        return example
    return tokenize_function

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

def print_datasets_shapes(tokenized_datasets):
    print(f"Shapes of the datasets:")
    print(f"Training: {tokenized_datasets['train'].shape}")
    print(f"Validation: {tokenized_datasets['validation'].shape}")
    print(f"Test: {tokenized_datasets['test'].shape}")

def get_model_output(tokenizer, model, prompt):

    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"], 
            max_new_tokens=200,
        )[0], 
        skip_special_tokens=True
    )
    return output

def print_output_comparison(prompt, output, output_new):
    dash_line = '-'.join('' for x in range(100))
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'Old model output:\n{output}\n')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output_new}')

def get_prompt_dialog_from_dataset(dataset, index):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    prompt = f"""
    Summarize the following conversation.

    {dialogue}

    Summary:
    """
    return prompt

def training_orig_model(model, tokenizer, dataset):
    tokenize_function = tokenizer_factory(tokenizer)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])
    #tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)
    training_args = TrainingArguments(
        output_dir="/tmp/",
        learning_rate=1e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=1,
        max_steps=3
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation']
    )
    trainer.train()

def get_tokenized_datasets(tokenizer, dataset):
    tokenize_function = tokenizer_factory(tokenizer)

    # The dataset actually contains 3 diff splits: train, validation, test.
    # The tokenize_function code is handling all data across all splits in batches.
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])
    tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 95 == 0, with_indices=True)
    print(tokenized_datasets)
    return tokenized_datasets

#print_datasets_shapes(tokenized_datasets)
#print(tokenized_datasets)
#print(print_number_of_trainable_model_parameters(original_model))