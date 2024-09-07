from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

os.environ['TRANSFORMERS_CACHE'] = '/mnt/sdcard/hub'

# 모델과 토크나이저 로드
torch.random.manual_seed(0)
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

text = "hello, sample text"
inputs = tokenizer(text, return_tensors="pt")

# 입력 IDs와 위치 IDs 생성
input_ids = inputs['input_ids']
position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0)

# 임베딩 레이어 호출
embedded_tokens = model.get_input_embeddings()(input_ids)

# 초기 출력값을 임베딩 결과로 설정
current_output = embedded_tokens

# 0번부터 39번까지의 레이어를 순차적으로 호출
for i in range(40):  # 총 40개의 레이어 (0번부터 39번까지)
    # 레이어 호출
    layer_output = model.model.layers[i](current_output, position_ids=position_ids)
    
    # tuple에서 텐서 데이터만 추출
    if isinstance(layer_output, tuple):
        current_output = layer_output[0]  # 첫 번째 요소만 사용
    else:
        current_output = layer_output  # 텐서인 경우 그대로 사용

    print(f"{i}번째 레이어 출력:", current_output)

'''
# 최종 레이어 출력에 대해 레이어 정규화 적용
normalized_output = model.model.norm(current_output)
print("norm: ", normalized_output)

# 언어 모델링 헤드를 통해 최종 로짓 계산
logits = model.lm_head(normalized_output)

# 결과 출력 (토큰별 확률 분포)
print("최종 로짓 출력:", logits)

# 로짓을 사용하여 토큰을 텍스트로 변환
predicted_tokens = torch.argmax(logits, dim=-1)
generated_text = tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
print("생성된 텍스트:", generated_text)
'''

# 모델의 구조를 탐색하여 숨겨진 함수들을 확인합니다.
print("모델의 전체 구조:")
print(model)

# model.model 객체의 모든 속성(함수와 레이어) 출력
print("모델의 내부 함수와 속성:")
print(dir(model.model))

# 마지막 레이어 호출 (39번 레이어)
# layer_output = model.model.layers[39](current_output)
layer_output = current_output

# 숨겨진 함수 호출 예시
# 예를 들어, 'hidden_function'이 'model.model' 내부에 있다고 가정하고 호출합니다.
# 실제 함수 이름은 위의 `dir()` 출력을 통해 확인해야 합니다.
if hasattr(model.model, 'hidden_function'):
    hidden_output = model.model.hidden_function(layer_output)
    print("숨겨진 함수 출력:", hidden_output)

# norm 레이어 적용
normalized_output = model.model.norm(hidden_output if 'hidden_output' in locals() else layer_output)

# lm_head 적용
logits = model.lm_head(normalized_output)
print("최종 로짓 출력:", logits)