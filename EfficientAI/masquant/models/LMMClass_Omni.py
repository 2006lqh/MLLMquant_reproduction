# coding=utf-8 
# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import transformers
import torch
from .models_utils import BaseLM, find_layers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pdb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModel
)
from lmms_eval.models.simple.qwen2_5_omni import Qwen2_5_Omni
from lmms_eval.api.model import lmms
from typing import List, Optional, Tuple, Union
from lmms_eval.api.instance import Instance
from lmms_eval import utils
from lmms_eval.models.model_utils.audio_processing import split_audio
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64
import librosa
import numpy as np
import soundfile as sf
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from moviepy import VideoFileClip
from PIL import Image
from tqdm import tqdm
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

class LMMClass(lmms):
    def __init__(self, model_path):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_path

        # 注意，我们只使用 omni 的 thinker 模块
        if 'Qwen2.5-Omni' in model_path:
            from transformers import Qwen2_5OmniForConditionalGeneration
            import copy
            kwargs = {"device_map": 'auto', 'enable_audio_output': False, 'attn_implementation': 'flash_attention_2'}
            self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **kwargs, torch_dtype=torch.float16)
            self.model = self._model.to('cuda')
            self._config = self._model.config
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
            self._tokenizer = self.processor.tokenizer
            self.tokenizer = self._tokenizer
            self.system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            
            self.batch_size_per_gpu = 1
            self.max_num_frames = 768
            self.vocab_size = self._tokenizer.vocab_size
            self.device_map = 'auto'
            self.use_cache = True
            
            self._model.eval()
        elif 'MiniCPM' in model_path:
            import copy
            kwargs = {"torch_dtype": 'auto', "device_map": 'auto', "trust_remote_code": True, "attn_implementation": "flash_attention_2"}
            model = AutoModel.from_pretrained(model_path, **kwargs)
            self.model_origin = copy.deepcopy(model)
            self.model_config_origin = model.config            
            self.model = model.llm
            processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print(f'we will use minicpm llm model only.')            
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, config=config, device_map='cpu',torch_dtype=torch.float16)
            self.seqlen = self.model.config.max_position_embeddings
            
            self.model_origin = self.model
            self.model_config_origin = self.model.config            
        
        print("vocab size: ", self.vocab_size)
        
        self.use_custom_video_loader = False

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.seqlen

    @property
    def max_gen_toks(self):
        print("max_gen_toks fn")
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():

            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
    
    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_Omni")
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        current_use_audio = False  # Flag to check whether we are using video or not

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            # For better performance, please visit the Qwen-Omni repo to get the latest system prompt based on tasks.
            # https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks
            message = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
            for i, context in enumerate(contexts):
                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        current_use_audio = self._check_if_video_has_audio(visual)
                        if self.use_custom_video_loader:
                            visual = read_video_pyav_base64(visual, num_frm=self.max_num_frames, fps=self.fps, img_format="JPEG", max_image_size=self.max_image_size)
                            image_contents = list(map(lambda x: f"data:image/jpeg;base64,{x}", visual))
                            message.append({"role": "user", "content": [{"type": "video", "video": image_contents}, {"type": "text", "text": context}]})
                        else:  # Model video loader
                            message.append({"role": "user", "content": [{"type": "video", "video": visual}, {"type": "text", "text": context}]})

                    elif isinstance(visual, Image.Image):  # Single image
                        message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})

                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        single_message = {"role": "user", "content": []}
                        for v in visual:
                            single_message["content"].append({"type": "image", "image": v})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)

                    # Fixed code for audio messages
                    elif isinstance(visual, dict):  # Single audio
                        current_use_audio = True
                        audio = self.resample_audio(visual["array"], visual["sampling_rate"])
                        audio_splits = split_audio(audio, 4800000)  # Split the audio to 5 min chunks
                        single_message = {"role": "user", "content": []}
                        for i in range(len(audio_splits)):
                            single_message["content"].append({"type": "audio", "audio": audio_splits[i]})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)

                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, dict) for v in visual):  # Multiple audios
                        current_use_audio = True
                        for i, v in enumerate(visual):
                            audio = self.resample_audio(v["array"], v["sampling_rate"])
                            audio_splits = split_audio(audio, 4800000)  # Split the audio to 5 min chunks
                            single_message = {"role": "user", "content": []}
                            for j in range(len(audio_splits)):
                                single_message["content"].append({"type": "audio", "audio": audio_splits[j]})
                            single_message["content"].append({"type": "text", "text": context})
                            message.append(single_message)

                    else:
                        raise ValueError(f"Unknown visual type: {type(visual)}")

            text = self.processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(message, use_audio_in_video=current_use_audio)
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=current_use_audio)

            self.model = self.model.to(torch.bfloat16)

            if self.device_map == "auto":
                inputs = inputs.to("cuda").to(self.model.dtype)
            else:
                inputs = inputs.to(self.model.device).to(self.model.dtype)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            # try:
            if True:
                cont = self.model.generate(
                    **inputs,
                    return_audio=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    # max_new_tokens=gen_kwargs["max_new_tokens"],
                    max_new_tokens=2,
                    use_cache=self.use_cache,
                    use_audio_in_video=current_use_audio,
                    thinker_do_sample=False,
                )
            # except Exception as e:
            #     eval_logger.error(f"Error {e} in generating")
            #     answer = ""
            #     res.append(answer)
            #     pbar.update(1)
            #     self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
            #     continue

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                answers[i] = ans
            content = []
            for ans, context in zip(answers, contexts):
                res.append(ans)
                content.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
