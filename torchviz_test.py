from torchviz import make_dot
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

messages = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# 입력을 토큰화하여 텐서로 변환
input_texts = [msg["content"] for msg in messages]
inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

# 모델에 입력 전달하여 출력 생성
output = model(**inputs)

# Torchviz를 사용하여 모델의 계산 그래프 시각화
dot = make_dot(output.logits, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

# 그래프를 이미지로 저장
dot.render(filename='model_graph', format='png')

'''
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)


#torchviz & save img
make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
make_dot.render(filename='model_graph', format='png')
'''
