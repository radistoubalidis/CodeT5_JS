import sys
import torch
from train import CodeT5
import transformers
import lightning.pytorch as pl

MODEL_NAME = 'Salesforce/codet5-base'
CPKT_FILE = 'checkpoints/best-checkpoint.ckpt'



snippets = """
function(c, key) {
  let s1 = this.value.slice(0, this.cursor);
  let s2 = this.value.slice(this.cursor);
  this.value = `${\s1}${\c}${\s2}`;
  this.red = false;
  this.cursor = this.placeholder ? 0 : s1.length+1;
  this.render();
}
"""

def preprocess_single_sample(input_texts):
    tokenizer = transformers.RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    # Tokenize the input text
    tokenized_input = tokenizer(
        input_texts,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    
    return tokenized_input


X = preprocess_single_sample(snippets)
input_ids = X['input_ids']
att = X['attention_mask']
decoder_input_ids = X['input_ids']
model = CodeT5.load_from_checkpoint(CPKT_FILE)
model.to(torch.device('cpu'))
model.eval()
with torch.no_grad():
    predictions = model(input_ids=input_ids,attention_mask=att,labels=decoder_input_ids)

generated_text = predictions[0]  # Selecting the first sequence in the batch
generated_text_tokens = torch.argmax(generated_text, dim=-1)  # Convert logits to token IDs
generated_text_tokens_list = generated_text_tokens.tolist()  # Convert to a Python list

# Decode token IDs back to text using your tokenizer's decode function
tokenizer = transformers.RobertaTokenizer.from_pretrained(MODEL_NAME)
generated_text = tokenizer.decode(generated_text_tokens_list, skip_special_tokens=True)

print("Generated Text:", generated_text)