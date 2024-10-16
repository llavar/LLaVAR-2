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
parser.add_argument('--output_file', type=str, required=True, help='Output file to save results')
parser.add_argument('--img_path', type=str, required=True,help='Input base image path')
args = parser.parse_args()
def save_dict_to_json(dict_data, filename):
    with open(filename, 'a') as file:
        json.dump(dict_data, file)
        file.write('\n')
log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')
cache_dir="/model/llava_phi3.5_v"
model_id = "microsoft/Phi-3.5-vision-instruct" 
model = AutoModelForCausalLM.from_pretrained(model_id,cache_dir=cache_dir, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2') # use _attn_implementation='eager' to disable flash attention
processor = AutoProcessor.from_pretrained(model_id,cache_dir=cache_dir, trust_remote_code=True) 

def process_2pair(ques,ans,r_ques,r_ans,processor):
    ques=ques.replace("\n<image>", "").replace("<image>\n", "")
    messages1 = [ 
        {"role": "user", "content": f"<|image_1|>\n{ques}"}, 
        {"role": "assistant", "content": f"{ans}"},
        {"role": "user", "content": f"{r_ques}"}, 
        {"role": "assistant", "content": f"{r_ans}"}
    ] 
    prompt1 = processor.tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
    messages2 =  [ 
        {"role": "user", "content": f"<|image_1|>\n{ques}"}, 
        {"role": "assistant", "content": f"{ans}"},
        {"role": "user", "content": ""},
    ] 
    prompt2 = processor.tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
    if prompt2.endswith("<|assistant|>\n"):
        prompt2 = prompt2[:-14]
    messages3 =  [ 
        {"role": "user", "content": f"<|image_1|>\n{r_ques}"}, 
        {"role": "assistant", "content": f"{r_ans}"}
    ]
    prompt3 = processor.tokenizer.apply_chat_template(messages3, tokenize=False, add_generation_prompt=True)
    messages4 = [ 
        {"role": "user", "content": f"<|image_1|>\n"}
    ]
    prompt4 = processor.tokenizer.apply_chat_template(messages4, tokenize=False, add_generation_prompt=True)
    input_ids1=processor.tokenizer(prompt1)['input_ids']
    input_ids2=processor.tokenizer(prompt2)['input_ids']
    input_ids3=processor.tokenizer(prompt3)['input_ids']
    input_ids4=processor.tokenizer(prompt4)['input_ids']
    return prompt1,prompt2,prompt3,prompt4,len(input_ids1),len(input_ids2),len(input_ids3),len(input_ids4)

f = open(args.input_file) #json
data = json.load(f)
for ind in data:
    image_path=args.img_path+ f"{ind['image']}"
    try:
        image = Image.open(image_path).convert('RGB')
    except:
        print(ind['id'],':fail')
        continue
    convs=ind['conversations']
    for i in range(0, len(convs), 4): 
        if (ind['id'],i) not in []:
            temp_data_i = {}
            extractive_question= ind['conversations'][i]['value']
            extractive_answer= ind['conversations'][i+1]['value']
            self_explain_question= ind['conversations'][i+2]['value']
            self_explain_answer= ind['conversations'][i+3]['value']
            
            p1,p2,p3,p4,l1,l2,l3,l4=process_2pair(extractive_question,extractive_answer,self_explain_question,self_explain_answer,processor)
            
            inputs = processor(x, [image], return_tensors="pt").to("cuda")
            target_ids1 = inputs['input_ids'].clone()
            whole_len1=target_ids1.size(1)

            start_token1=whole_len1-(l1-(l2-3))
            target_ids1[:, :whole_len1-(l1-(l2-3))] = -100

            inputs_ans=processor(p3, [image], return_tensors="pt").to("cuda") 
            target_ids2 = inputs_ans['input_ids'].clone()
            whole_len2=target_ids2.size(1)

            start_token2=whole_len2-(l3-(l4-4))
            target_ids2[:, :whole_len2-(l3-(l4-4))] = -100

            #conditioned
            with torch.no_grad():
                outputs1 = model(inputs['input_ids'],labels=target_ids1,pixel_values=inputs['pixel_values'],image_sizes=inputs['image_sizes'])
                neg_log_likelihood1 = outputs1.loss
            #ans
            with torch.no_grad():
                outputs2 = model(inputs_ans['input_ids'],labels=target_ids2,pixel_values=inputs['pixel_values'],image_sizes=inputs['image_sizes'])
                neg_log_likelihood2 = outputs2.loss

            losses1 = []
            logits1 = outputs1.logits
            for j in range(1, whole_len1):
                log_prob_dist1 = log_softmax(logits1[0, j-1])
                true_token1 = inputs['input_ids'][0, j]
                token_loss1 = nll_loss(log_prob_dist1.unsqueeze(0),  torch.where(true_token1 == -1, torch.tensor(-100, device='cuda'), true_token1).unsqueeze(0))
                losses1.append(token_loss1.item())
            self_explain_cond_extract_img=np.array(losses1[start_token1-1:whole_len1-1]).mean()
            
            losses2 = []
            logits2 = outputs2.logits
            for j in range(1, whole_len2):
                log_prob_dist2 = log_softmax(logits2[0, j-1])
                true_token2 = inputs_ans['input_ids'][0, j]
                token_loss2 = nll_loss(log_prob_dist2.unsqueeze(0), torch.where(true_token2 == -1, torch.tensor(-100, device='cuda'), true_token2).unsqueeze(0))
                losses2.append(token_loss2.item())
            self_explain_cond_img=np.array(losses2[start_token2-1:whole_len2-1]).mean()

            efd=self_explain_cond_extract_img/self_explain_cond_img
            temp_data_i['id']=ind['id']
            temp_data_i['qa_id']=i
            temp_data_i['efd_score']=[self_explain_cond_extract_img,self_explain_cond_img,efd]
            save_dict_to_json(temp_data_i,args.output_file) #ndjson