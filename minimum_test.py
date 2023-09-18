from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer

mymodelname = 'google/flan-t5-base'
mymodel = AutoModelForSeq2SeqLM.from_pretrained(mymodelname)
mytokenizer = AutoTokenizer.from_pretrained(mymodelname, use_fast=True)
myprompt = "Who is Pedro Melendez? please explain in 20 words or more"
myinput = mytokenizer(myprompt, return_tensors='pt')
myoutput = mytokenizer.decode(mymodel.generate(myinput["input_ids"], max_new_tokens=50)[0], skip_special_tokens=True)
print(myoutput)