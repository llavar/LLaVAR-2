from PIL import Image 
import requests 
import json
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
import torch
import torch.nn as nn
import numpy as np
import json
import re
import argparse
parser = argparse.ArgumentParser(description="Run model inference with external inputs.")
parser.add_argument('--input_file', type=str, required=True, help='Input JSON file with data')
parser.add_argument('--img_path', type=str, required=True,help='Input base image path')
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
cache_dir="/model/llava_phi3.5_v"
model_id = "microsoft/Phi-3.5-vision-instruct" 

model = AutoModelForCausalLM.from_pretrained(model_id,cache_dir=cache_dir, trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2').to(device) # use _attn_implementation='eager' to disable flash attention
processor = AutoProcessor.from_pretrained(model_id,cache_dir=cache_dir, trust_remote_code=True) 

def process_1pair(ques,ans,processor):
    ques=ques.replace("\n<image>", "").replace("<image>\n", "")
    messages1 = [ 
        {"role": "user", "content": f"<|image_1|>\n{ques}"}, 
        {"role": "assistant", "content": f"{ans}"}
    ] 
    prompt1 = processor.tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=False)
    messages2 =  [ 
        {"role": "user", "content": f"{ques}"}, 
        {"role": "assistant", "content": f"{ans}"}
    ] 
    prompt2 = processor.tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=False)
    input_ids1=processor.tokenizer(prompt1)['input_ids']
    input_ids2=processor.tokenizer(prompt2)['input_ids']
    return prompt1,prompt2,len(input_ids1),len(input_ids2)

def save_ndjson(list_of_dicts, filename):
    with open(filename, 'w') as file:  
        for dict_data in list_of_dicts:
            json.dump(dict_data, file)
            file.write('\n')
            
f = open(args.input_file)#json
data = json.load(f)
all_results=[]
for ind in data:
    image_path=args.img_path+ f"{ind['image']}"
    try:
        image = Image.open(image_path).convert('RGB')
    except:
        print(ind['id'],':fail')
        continue
    convs=ind['conversations']
    for i in range(0, len(convs), 2): 
        if (ind['id'],i) not in []:
            temp_data_i = {}
            extractive_question= ind['conversations'][i]['value']
            extractive_answer= ind['conversations'][i+1]['value']
            p1,p2,l1,l2=process_1pair(extractive_question,extractive_answer,processor)

            inputs = processor(p1,  [image],return_tensors="pt").to(device)
            target_ids1 = inputs['input_ids'].clone()
            whole_len1=target_ids1.size(1)
            start_token1=whole_len1-l1+12
            target_ids1[:, :whole_len1-l1+12] = -100

            inputs_ans=processor(p2,return_tensors="pt").to(device) 
            target_ids2 = inputs_ans['input_ids'].clone()
            whole_len2=target_ids2.size(1)
            start_token2=4
            target_ids2[:, :4] = -100

            with torch.no_grad():
                outputs1 = model(inputs['input_ids'],labels=target_ids1.contiguous(),pixel_values=inputs['pixel_values'],image_sizes=inputs['image_sizes'])
                neg_log_likelihood1 = outputs1.loss

            with torch.no_grad():
                outputs2 = model(inputs_ans['input_ids'],labels=target_ids2)
                neg_log_likelihood2 = outputs2.loss

            losses1 = []
            logits1 = outputs1.logits
            for j in range(1, whole_len1):
                log_prob_dist1 = log_softmax(logits1[0, j-1])
                true_token1 = inputs['input_ids'][0, j]
                token_loss1 = nll_loss(log_prob_dist1.unsqueeze(0),  torch.where(true_token1 == -1, torch.tensor(-100, device=device), true_token1).unsqueeze(0))
                losses1.append(token_loss1.item())
            qa_cond_img=np.array(losses1[start_token1-1:whole_len1-1]).mean()
            
            losses2 = []
            logits2 = outputs2.logits
            for j in range(1, whole_len2):
                log_prob_dist2 = log_softmax(logits2[0, j-1])
                true_token2 = inputs_ans['input_ids'][0, j]
                token_loss2 = nll_loss(log_prob_dist2.unsqueeze(0), torch.where(true_token2 == -1, torch.tensor(-100, device=device), true_token2).unsqueeze(0))
                losses2.append(token_loss2.item())
            qa_no_cond=np.array(losses2[start_token2-1:whole_len2-1]).mean()
            vfd=qa_cond_img/qa_no_cond
        
            temp_data_i['id']=ind['id']
            temp_data_i['qa_id']=i
            temp_data_i['vfd_score']=[qa_cond_img,qa_no_cond,vfd]
            all_results.append(temp_data_i)  

save_ndjson(all_results, args.output_file)




