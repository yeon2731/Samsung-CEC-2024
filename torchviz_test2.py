from torchviz import make_dot
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 재현성을 위해 시드 설정
torch.manual_seed(0)

# 모델과 토크나이저 로드
model_id = "microsoft/Phi-3-medium-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 입력 메시지 정의
messages = [
    "Can you provide ways to eat combinations of bananas and dragonfruits?",
    "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."
]

# 메시지를 텐서로 변환
inputs = tokenizer(messages, return_tensors="pt", padding=True, truncation=True)

# 텐서를 CUDA 장치로 이동 (GPU 사용 시)
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# 모델에 텐서 입력
with torch.no_grad():
    outputs = model(**inputs)


make_dot(outputs, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
make_dot.render(filename='model_graph', format='png')

