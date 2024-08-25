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
inputs = {key: value.to("cuda") for key, value in tokenizer(prompt, return_tensors="pt").items()}

# 모델을 ONNX 형식으로 내보내기
onnx_model_path = "phi3_model.onnx"
torch.onnx.export(
    model,
    (inputs["input_ids"],),  # 입력 텐서
    onnx_model_path,          # 저장할 ONNX 모델 파일 경로
    opset_version=14,         # ONNX opset 버전 (14 이상 사용해야 함)
    input_names=["input_ids"],  # 입력 텐서 이름
    output_names=["output"],   # 출력 텐서 이름
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},  # 배치 크기와 시퀀스 길이를 동적으로 설정
        "output": {0: "batch_size", 1: "sequence_length"},
    },
)
