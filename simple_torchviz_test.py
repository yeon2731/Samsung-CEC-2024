import torch
import torch.nn as nn
from torchviz import make_dot

# 간단한 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# 모델과 입력 데이터 생성
model = SimpleModel()
input_data = torch.randn(1, 10)

# 모델에 입력 데이터를 통과시켜서 출력 생성
output = model(input_data)

# 그래프 시각화
dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

# 그래프를 PNG 파일로 저장
dot.render(filename='model_graph', format='png')  # 'model_graph.png' 파일로 저장




