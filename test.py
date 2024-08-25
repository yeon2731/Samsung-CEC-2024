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

messages = [ {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"}, ]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

#output = pipe(messages, **generation_args)

input_text = messages[0]["content"]
encoding = tokenizer(input_text, return_tensors="pt")

# 인코딩된 데이터는 이미 텐서로 되어 있음
input_ids = encoding["input_ids"].to("cuda")
attention_mask = encoding["attention_mask"].to("cuda")

# 모델을 ONNX 형식으로 변환
torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "model.onnx",
    opset_version=14,
    input_names=['input_ids', 'attention_mask'],
    output_names=['output'],
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                  'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                  'output': {0: 'batch_size', 1: 'sequence_length'}}
)
