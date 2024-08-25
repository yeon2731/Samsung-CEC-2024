import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)
model_id = "microsoft/Phi-3-medium-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "This is an example script"
#inputs = tokenizer(prompt, return_tensors="pt")
inputs = {key: value.to("cuda") for key, value in tokenizer(prompt, reuturn_tensors="pt").items()}


generate_ids = model.generate(inputs["input_ids"], max_length=30)
decode_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(decode_text)
