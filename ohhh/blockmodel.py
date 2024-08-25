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

class DecoderLayerBlock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.self_attn = layer.self_attn
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.resid_attn_dropout = layer.resid_attn_dropout
        self.resid_mlp_dropout = layer.resid_mlp_dropout
        self.post_attention_layernorm = layer.post_attention_layernorm

    def forward(self, x, position_ids=None):
        print("DecoderLayerBlock")
        residual = x
        x = self.input_layernorm(x)
            
        outputs = self.self_attn(x, position_ids=position_ids)
        #print(outputs)

        x, _, _ = self.self_attn(x, position_ids=position_ids)

        x = self.resid_attn_dropout(x)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = self.resid_mlp_dropout(x)
        x = x + residual

        return x

class NormalizationBlock(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm = model.model.norm

    def forward(self, x):
        print("NormalizationBlock")
        return self.norm(x)

class OutputBlock(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm_head = model.lm_head

    def forward(self, x):
        print("OutputBlock")
        logits = self.lm_head(x)
        token_ids = torch.argmax(logits, dim=-1)
        return token_ids

# 모델 블록 클래스 정의
class Phi3ForCausalLM_Blockwise(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.device = model.device
                                    
        # Block 1: Embedding Block
        self.embed_tokens = model.model.embed_tokens
        self.embed_dropout = model.model.embed_dropout

        self.blocks = nn.ModuleList()

        # Blocks 2-41: Decoder Layer Blocks (40 blocks)
        for i in range(40):
            self.blocks.append(DecoderLayerBlock(model.model.layers[i]))
                                        
        # Block 42: Normalization Block
        self.blocks.append(NormalizationBlock(model))
                                                                
        # Block 43: Output Block
        self.blocks.append(OutputBlock(model))

    def forward(self, input_ids):
        x = input_ids
        print(f"blockwise x.shape : {x.shape}")
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x

    def pipeline_process(self, inputs, batch_size=1):
        results = []
        data_loader = DataLoader(inputs, batch_size=batch_size, shuffle=False)
        
        for batch in data_loader:
            batch_results = []
            for x in batch:
                result = self._process_single_input(x)
                batch_results.append(result)
            results.extend(batch_results)
                                                                                                                    
        return results

    def _process_single_input(self, input_ids):

        batch_size, seq_len = input_ids.size(0), input_ids.size(1)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
                    

        x = input_ids.to(self.device)

        x = self.embed_tokens(x)
        x = self.embed_dropout(x)

        for i, block in enumerate(self.blocks):
            if hasattr(block, 'self_attn'):
                x = block(x, position_ids=position_ids)
            else:
                x = block(x)
            print(f"Block {i+1} after shape :{x.shape}")
        return x

model_id = "microsoft/Phi-3-medium-4k-instruct" # please replace with local model path
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
 
model = Phi3ForCausalLM_Blockwise(model)

"""

####### Section 2. GPU Warm up #######
messages = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

input_ids_list = [tokenizer(message["content"], return_tensors="pt",padding=True).input_ids.to(model.device) for message in messages]
outputs = model.pipeline_process(input_ids_list)
decoded_outputs = [tokenizer.decode(output[0].tolist(), skip_special_tokens=True) for output in outputs]
print(decoded_outputs[0])

print("section 2 end")

"""

####### Section 3. Load data and Inference -> Performance evaluation part #######
start = time.time()
data = load_dataset("json", data_files="test_dataset.jsonl")['train']

print(type(data['message']))
print(data['message'][0])  # 첫 번째 요소를 확인

# 메시지를 토큰화하여 input_ids 리스트로 변환
input_ids_list = [
    tokenizer(message["content"], return_tensors="pt").input_ids.to(model.device)
    for message in data['message']
]

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

