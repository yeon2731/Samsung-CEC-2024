import time
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from torch.utils.data import DataLoader

####### Section 1. Set up #######
torch.random.manual_seed(0)

class EmbeddingBlock(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer

    def forward(self, input_ids):
        return self.embedding_layer(input_ids)

class DecoderBlock(nn.Module):
    def __init__(self, decoder_layer):
        super().__init__()
        self.decoder_layer = decoder_layer

    def forward(self, x):
        return self.decoder_layer(x)

class NormalizationBlock(nn.Module):
    def __init__(self, normalization_layer):
        super().__init__()
        self.normalization_layer = normalization_layer
    
    def forward(self, x):
        return self.normalization_layer(x)

class OutputBlock(nn.Module):
    def __init__(self, output_layer):
        super().__init__()
        self.output_layer = output_layer

    def forward(self, x):
        logits = self.output_layer(x)
        token_ids = torch.argmax(logits, dim=-1)
        return token_ids

# 모델 블록 클래스 정의
class Phi3ForCausalLM_Blockwise(nn.Module):
    def __init__(self, embedding_layer, decoder_layers, normalization_layer, output_layer):
        super().__init__()
        self.device = device
        self.embedding_block = EmbeddingBlock(embedding_layer)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(layer) for layer in decoder_layers])
        self.normalization_block = NormalizationBlock(normalization_layer)
        self.output_block = OutputBlock(output_layer)

    def forward(self, input_ids, position_ids=None):
        x = self.embedding_block(input_ids.to(self.device))
        for block in self.decoder_blocks:
            x = block(x)
        x = self.normalization_block(x)
        x = self.output_block(x)
        return x


model_id = "microsoft/Phi-3-medium-4k-instruct" # please replace with local model path
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
 
embedding_layer = model.model.embed_tokens
decoder_layers = model.model.layers
normalization_layer = model.model.norm
output_layer = model.lm_head

#model = Phi3ForCausalLM_Blockwise(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Phi3ForCausalLM_Blockwise(embedding_layer, decoder_layers, normalization_layer, output_layer)


####### Section 2. GPU Warm up #######
messages = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

input_ids_list = [tokenizer(message["content"], return_tensors="pt",padding=True).input_ids.to(model.device) for message in messages]

position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=model.device).unsqueeze(0).expand(input_ids.size(0), -1)


decoded_outputs = []
for input_ids in input_ids_list:
    output = model(input_ids, position_ids=position_ids)
    decoded_output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    decoded_outputs.append(decoded_output)

#input_ids = torch.cat(input_ids_list, dim=0)
#outputs = model(input_ids)
#decoded_outputs = [tokenizer.decode(output[0].tolist(), skip_special_tokens=True) for output in outputs]
print(decoded_outputs[0])

print("section 2 end")

"""

####### Section 3. Load data and Inference -> Performance evaluation part #######
start = time.time()
data = load_dataset("json", data_files="test_dataset.jsonl")['train']

# 메시지를 토큰화하여 input_ids 리스트로 변환
input_ids_list = [tokenizer(message, return_tensors="pt").input_ids.to(model.device) for message in data['message']]
# 파이프라인 병렬 처리 실행
batch_size = 16
outs = model.pipeline_process(input_ids_list, batch_size=batch_size)
# 출력 결과를 디코딩하여 텍스트로 변환
decoded_outputs = [tokenizer.decode(output[0], skip_special_tokens=True) for output in outs]

end = time.time()

####### Section 4. Accuracy (Just for leasderboard) #######
print("===== Answers =====")
correct = 0
for i, out in enumerate(decoded_outputs):
    correct_answer = data[i]["answer"]
    answer = out.lstrip().replace("\n","")
    if answer == correct_answer:
        correct += 1
    print(answer)
                         
print("===== Perf result =====")
print("Elapsed_time: ", end-start)
print(f"Correctness: {correct}/{len(data)}")

"""
