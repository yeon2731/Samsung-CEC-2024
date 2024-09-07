from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

os.environ['TRANSFORMERS_CACHE'] = '/home/jetson/.cache/hub'

# 모델과 토크나이저 로드
model_name = "microsoft/Phi-3-medium-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
print("Finish to load model")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Finish to load tokenizer")

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델을 GPU로 이동
model = model.to(device)
print("Complete to move model to GPU")

# 입력 텍스트를 정의하고 토큰화
input_text = "Hello, how can I help you today?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# 0번 레이어 (첫 번째 레이어) 가져오기
first_layer = model.transformer.h[0]

# 첫 번째 레이어 실행
with torch.no_grad():  # 그래디언트 계산 비활성화 (메모리 최적화)
    layer_output = first_layer(inputs.input_ids)

# 출력 결과를 확인
print("First layer output:", layer_output)

# 1번 레이어 (두 번째 레이어) 가져오기
second_layer = model.transformer.h[1]

# 두 번째 레이어 실행
with torch.no_grad():
    second_layer_output = second_layer(layer_output[0])  # 첫 번째 레이어의 출력 결과를 두 번째 레이어에 입력

# 출력 결과를 확인
print("Second layer output:", second_layer_output)

