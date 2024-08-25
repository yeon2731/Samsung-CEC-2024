import torch
#import torch._dynamo as dynamo
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
 
 ####### Section 1. Set up #######
torch.random.manual_seed(0)
model_id = "microsoft/Phi-3-medium-4k-instruct" # please replace with local model path
model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(model)
