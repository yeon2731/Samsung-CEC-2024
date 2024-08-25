from transformers import AutoImageProcessor, ResNetForImageClassification

import torchvision.models as models
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
#resnet50 = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(image, return_tensors="pt")

resnet50 = models.resnet50(pretrained=True)

class ConvBlock(torch.nn.Module):
    def __init__(self, layers):
        super(ConvBlock, self).__init__()
        self.block = torch.nn.Sequential(*layers)
    def forward(self,x):
        return self.block(x)

conv1_block = ConvBlock([
    resnet50.conv1,
    resnet50.bn1,
    resnet50.relu,
    resnet50.maxpool
    ])

# Conv2_x 블록
conv2_block = ConvBlock([
    resnet50.layer1
])

# Conv3_x 블록
conv3_block = ConvBlock([
    resnet50.layer2
])

# Conv4_x 블록
conv4_block = ConvBlock([
    resnet50.layer3
])

# Conv5_x 블록
conv5_block = ConvBlock([
    resnet50.layer4
])

# Average Pooling 및 Fully Connected Layer
avg_pool_fc = ConvBlock([
    resnet50.avgpool,
    torch.nn.Flatten(),  # 2D 출력 -> 1D 벡터 변환
    resnet50.fc
])


    # 블록별 인퍼런스 수행
def inference_step_by_step(image):
    x = conv1_block(image)
    print(x.shape)
    x = conv2_block(x)
    print(x.shape)
    x = conv3_block(x)
    print(x.shape)
    x = conv4_block(x)
    print(x.shape)
    x = conv5_block(x)
    print(x.shape)
    output = avg_pool_fc(x)
    return output

# 예시 인퍼런스
image = torch.randn(1, 3, 224, 224)  # 더미 입력 이미지
output = inference_step_by_step(image)
print(output.shape)


