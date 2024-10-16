from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor ,AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
import json
import re
import os
from tqdm import tqdm 
import argparse
parser = argparse.ArgumentParser(description="Run model inference with external inputs.")
parser.add_argument('--input_file', type=str, required=True, help='Input JSON file with data')
parser.add_argument('--output_file', type=str, required=True, help='Output file to save results')
parser.add_argument('--device', type=int, required=True, help='CUDA device to use')
args = parser.parse_args()
device=f"cuda:{args.device}"
def save_dict_to_json(dict_data, filename):
    with open(filename, 'a') as file:
        json.dump(dict_data, file)
        file.write('\n')
log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')
cache_dir="/model/llava_phi3"
model_id ="microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id,cache_dir=cache_dir, trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2').to(device) # use _attn_implementation='eager' to disable flash attention
tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir, trust_remote_code=True) 
def process_2pair(ques,ans,processor):
    ques=ques.replace("\n<image>", "").replace("<image>\n", "")
    messages1 = [ 
        {"role": "user", "content": f"{ques}"}, 
        {"role": "assistant", "content": f"{ans}"}
    ] 
    prompt1 = processor.apply_chat_template(messages1, tokenize=False, add_generation_prompt=False)

    messages2 =  [ 
        {"role": "assistant", "content": f"{ans}"}
    ] 
    prompt2 = processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=False)
    input_ids1=processor(prompt1,return_tensors="pt").to(device)
    input_ids2=processor(prompt2,return_tensors="pt").to(device)
    return prompt1,prompt2,input_ids1,input_ids2,len(input_ids1['input_ids']),len(input_ids2['input_ids'])
f = open(args.input_file)
data = json.load(f)
for ind in tqdm(data):
    convs=ind['conversations']
    for i in range(0, len(convs), 2): 
        if (ind['id'],i) not in []:
            temp_data_i = {}
            extractive_question= ind['conversations'][i]['value']
            extractive_answer= ind['conversations'][i+1]['value']
            p1,p2,in_id1,in_id2,l1,l2=process_2pair(extractive_question,extractive_answer,tokenizer)
            inputs = in_id1
            inputs2 = in_id2

            target_ids1 = inputs['input_ids'].clone()
            whole_len1=target_ids1.size(1)
            start_token1=l1-l2+1
            target_ids1[:, :-l2+1] = -100

            target_ids2 = inputs2['input_ids'].clone()
            start_token2=1
            target_ids2[:, :1] = -100
            whole_len2=target_ids2.size(1)
            #conditioned
            with torch.no_grad():
                outputs1 = model(inputs['input_ids'],labels=target_ids1)
                neg_log_likelihood1 = outputs1.loss
            #ans
            with torch.no_grad():
                outputs2 = model(inputs2['input_ids'],labels=target_ids2)
                neg_log_likelihood2 = outputs2.loss

            losses1 = []
            logits1 = outputs1.logits
            for j in range(1, whole_len1):
                log_prob_dist1 = log_softmax(logits1[0, j-1])
                true_token1 = inputs['input_ids'][0, j]
                token_loss1 = nll_loss(log_prob_dist1.unsqueeze(0),  torch.where(true_token1 == -1, torch.tensor(-100, device=device), true_token1).unsqueeze(0))
                losses1.append(token_loss1.item())
            mean1=np.array(losses1[start_token1-1:whole_len1-1]).mean()
            
            losses2 = []
            logits2 = outputs2.logits
            for j in range(1, whole_len2):
                log_prob_dist2 = log_softmax(logits2[0, j-1])
                true_token2 = inputs2['input_ids'][0, j]
                token_loss2 = nll_loss(log_prob_dist2.unsqueeze(0), torch.where(true_token2 == -1, torch.tensor(-100, device=device), true_token2).unsqueeze(0))
                losses2.append(token_loss2.item())
            mean2=np.array(losses2[start_token2-1:whole_len2-1]).mean()
            ifd=mean1/mean2
            temp_data_i['id']=ind['id']
            temp_data_i['qa_id']=i
            temp_data_i['ifd_score']=[mean1,mean2,ifd]
            save_dict_to_json(temp_data_i,args.output_file)


