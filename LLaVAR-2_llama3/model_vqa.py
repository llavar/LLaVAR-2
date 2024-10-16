import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from paddleocr import PaddleOCR, draw_ocr
import logging
import numpy as np
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from easyocr.utils import get_paragraph

from PIL import Image
import math
logger = logging.getLogger('ppocr')
logger.setLevel(logging.WARNING)

ppocr = PaddleOCR(use_angle_cls=True, lang='en')
def union_box(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    return [x1, y1, x2, y2]

def is_same_line(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    
    box1_midy = (box1[1] + box1[3]) / 2
    box2_midy = (box2[1] + box2[3]) / 2

    if box1_midy < box2[3] and box1_midy > box2[1] and box2_midy < box1[3] and box2_midy > box1[1]:
        return True
    else:
        return False


def is_adj_line(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    h1 = box1[3] - box1[1]
    h2 = box2[3] - box2[1]
    h_dist = max(box1[1], box2[1]) - min(box1[3], box2[3])

    box1_midx = (box1[0] + box1[2]) / 2
    box2_midx = (box2[0] + box2[2]) / 2

    # if h_dist <= min(h1, h2) and box1_midx < box2[2] and box1_midx > box2[0] and box2_midx < box1[2] and box2_midx > box1[0]:
    if h_dist <= min(h1, h2): # v2
        return True
    else:
        return False

def boxes_sort(boxes):
    """ From left top to right bottom
    Params:
        boxes: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    """
    sorted_id = sorted(range(len(boxes)), key=lambda x: (boxes[x][1], boxes[x][0]))
def reorg_texts(ocr):
    texts_set = []
    boxes_set = []
    for line in ocr:
        texts_set.append(line[1])
        boxes_set.append(line[0])
    boxes_set = [text_box[0]+text_box[2] for text_box in boxes_set]
    return '\n'.join(space_layout(texts_set, boxes_set))
def space_layout(texts, boxes):
    line_boxes = []
    line_texts = []
    max_line_char_num = 0
    line_width = 0
    # print(f"len_boxes: {len(boxes)}")
    while len(boxes) > 0:
        line_box = [boxes.pop(0)]
        line_text = [texts.pop(0)]
        char_num = len(line_text[-1])
        line_union_box = line_box[-1]
        while len(boxes) > 0 and is_same_line(line_box[-1], boxes[0]):
            line_box.append(boxes.pop(0))
            line_text.append(texts.pop(0))
            char_num += len(line_text[-1])
            line_union_box = union_box(line_union_box, line_box[-1])
        line_boxes.append(line_box)
        line_texts.append(line_text)
        if char_num >= max_line_char_num:
            max_line_char_num = char_num
            line_width = line_union_box[2] - line_union_box[0]
    
    # print(line_width)

    char_width = line_width / max_line_char_num
    # print(char_width)
    if char_width == 0:
        char_width = 1

    space_line_texts = []
    for i, line_box in enumerate(line_boxes):
        space_line_text = ""
        for j, box in enumerate(line_box):
            left_char_num = int(box[0] / char_width)
            space_line_text += " " + " " * int((left_char_num - len(space_line_text)) * 0.4)
            space_line_text += line_texts[i][j]
        space_line_texts.append(space_line_text)

    return space_line_texts

def get_ocr(img):
    result2 = ppocr.ocr(np.asarray(img), cls=True)[0]
    if result2 is not None:
        result2 = [(item[0], item[1][0], item[1][1]) for item in result2]
        p_result2 = get_paragraph(result2)
    else:
        p_result2 = []
    return p_result2, result2
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    with open(args.question_file, 'r') as file:
        questions = json.load(file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["id"]
        if idx<37562:
            continue
        image_file = line["image"]
        for k in range(0,len(line["conversations"]),2):
            qs=line["conversations"][k]['value'].replace("\n<image>", "").replace("<image>\n", "")
            cur_prompt = qs
            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
            ############################
            ocr_res, ocr = get_ocr(image)
            if ocr is not None:
                if len(ocr) > 4:
                    added = 'Given OCR-based Page Parser Results: ' + reorg_texts(ocr) + '\n'
                else:
                    added = 'OCR: ' + ' '.join([item[1] for item in ocr_res]) + "\n"
            else:
                added = ""
            qs = added + qs
            ############################
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()


            image_tensor = process_images([image], image_processor, model.config)[0]
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llama3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()
    eval_model(args)
