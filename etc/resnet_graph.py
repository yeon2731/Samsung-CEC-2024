from torchviz import make_dot
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from graphviz import Source
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
#predicted_label = logits.argmax(-1).item()
#print(model.config.id2label[predicted_label])

# 모델에 입력 전달하여 출력 생성
output = model(**inputs)

# Torchviz를 사용하여 모델의 계산 그래프 시각화
dot = make_dot(output.logits, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

# 그래프를 이미지로 저장
dot.render(filename='model_graph', format='pdf')
