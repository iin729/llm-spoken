import torch
import torch.distributed as dist
import torch.nn.functional as F
import re
import time
import warnings
from typing import List, Optional, Union

import transformers
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
"""
Zero_Ablation_Greedy_Search_Annotated.py
"""

def zero_ablation_greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
# (modified)added logits_warper args _sample call
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    """
     Zero Ablation Greedy Search
    update v3fix(Sticky Match)first
    """
    print(">>> DEBUG: Entering custom zero_ablation_greedy_search", flush=True)


    # ------------------ 1. initconfigargs ------------------
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated, use `stopping_criteria` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    # ======================================================================= #
    # (new fix) pad_token_id None after
    # ======================================================================= #
    if pad_token_id is None:
        if eos_token_id is not None and len(eos_token_id) > 0:
            pad_token_id = eos_token_id[0]
        else:
            pad_token_id = 0 #

    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.generation_config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    return_dict_in_generate = return_dict_in_generate if return_dict_in_generate is not None else self.generation_config.return_dict_in_generate

    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False
    count = 0

    # ------------------ 2. module ------------------
    # first model_kwargs self (fix)
    target_tokens_list_1 = model_kwargs.pop('hallucinated_tokens', getattr(self, 'hallucinated_tokens', []))
    target_tokens_list_2 = model_kwargs.pop('non_hallucinated_tokens', getattr(self, 'non_hallucinated_tokens', []))
    influence_score_type = model_kwargs.pop('influence_score', getattr(self, 'influence_score', 'prob_diff'))
    print(
        "DEBUG vocab sizes:",
        "hal=", len(target_tokens_list_1),
        "non_hal=", len(target_tokens_list_2),
        "score=", influence_score_type,
        flush=True
    )
    print("DEBUG hal examples:", list(target_tokens_list_1)[:20], flush=True)
    print("DEBUG non_hal examples:", list(target_tokens_list_2)[:20], flush=True)

    # (optimization 1)lengthfirst "that that" "that"
    # pass via list set list
    if target_tokens_list_1:
        target_tokens_list_1 = sorted(list(target_tokens_list_1), key=lambda x: len(x.split()), reverse=True)
    if target_tokens_list_2:
        target_tokens_list_2 = sorted(list(target_tokens_list_2), key=lambda x: len(x.split()), reverse=True)

    hallucination_influences = {}
    non_hallucination_influences = {}
    o_proj_modules = []

    # args
    num_hidden_layers = getattr(self.config, "num_hidden_layers", 32)
    num_attention_heads = getattr(self.config, "num_attention_heads", 32)
    hidden_size = getattr(self.config, "hidden_size", 4096)
    # head dim
    head_dim = hidden_size // num_attention_heads

    # tokenizer (pass via tokenizer)
    # local_tokenizer = model_kwargs.get("tokenizer", None)
    # local_tokenizer = model_kwargs.pop("tokenizer", None)

    # tokenizer
    # (fix)first kwargs self.zero_ablation_tokenizer
    local_tokenizer = model_kwargs.pop("tokenizer", getattr(self, 'zero_ablation_tokenizer', None))

    if local_tokenizer is None:
        # defaultor
        raise ValueError("Please pass 'tokenizer' in model_kwargs for Zero Ablation Search.")

    # o_proj
    for name, module in self.model.named_modules():
        if 'o_proj' in name:
            o_proj_modules.append(module)

    # andconfig
    if len(o_proj_modules) != num_hidden_layers:
        warnings.warn(f"Found {len(o_proj_modules)} o_proj modules but config says {num_hidden_layers} layers.")

    # ======================================================================= #
    # (new fix) Transformers 4.46+ Llama/Qwen etc. cache_position
    # ======================================================================= #
    if model_kwargs.get("cache_position") is None:
        # cache_position tensorcurrent
        # shape: (seq_len,)
        model_kwargs["cache_position"] = torch.arange(input_ids.shape[1], device=input_ids.device)


    # ------------------ 3. generateloop ------------------
    while True:
        if synced_gpus:
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # =======================================================
        # (fix)Mistral unexpected keyword argument 'generation_config'
        # remove generation_config
        # =======================================================
        model_inputs.pop("generation_config", None)

        #
        outputs = self(
            **model_inputs,
            return_dict=True,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]
        next_tokens_scores = logits_processor(input_ids, next_token_logits).detach()
        next_tokens = torch.argmax(next_tokens_scores, dim=-1).detach()

        # current token
        next_word = local_tokenizer.decode(next_tokens)

        original_probs = F.softmax(next_token_logits, dim=-1)
        original_log_probs = F.log_softmax(next_token_logits, dim=-1)

        # ------------------ 4. () ------------------

        def custom_hook(module, input, layer_idx, head_idx):
            ablated_input = input[0]
            # head_dim
            start_idx = head_dim * head_idx
            end_idx = head_dim * (head_idx + 1)
            ablated_input[:, :, start_idx: end_idx] = 0
            return (ablated_input,)

        def attach_custom_hook(layer_idx, head_idx):
            return lambda m, i: custom_hook(m, i, layer_idx, head_idx)

        #
        lookback_len = 25
        current_seq_ids = torch.cat([input_ids[0][-lookback_len:], next_tokens[0].unsqueeze(0)], dim=0)
        current_text_window = local_tokenizer.decode(current_seq_ids, skip_special_tokens=True)
        # print 60 log
        if count < 60:
            print(f"DEBUG step={count} next_word={repr(next_word)}", flush=True)
            print(f"DEBUG step={count} window={repr(current_text_window)}", flush=True)

        # modify 2 current_gen_token
        def is_valid_match(text_window, targets, current_gen_token_str):
            """
            function
            1. currentgenerate token (next_word) ifskip
            2. text_window
            3. targets /
            """
            # --- fix Sticky Match ---
            # ifcurrentgenerate token return
            # generate "um" after "um"
            if not re.search(r"[a-zA-Z0-9]", current_gen_token_str):
                return None, False

            # 1. "OK, you know, you" -> "ok you know you"
            # generate
            window_tokens = re.findall(r"[a-zA-Z0-9']+", text_window.lower())

            # ifreturn
            if not window_tokens:
                return None, False

            # token liststring endswith
            # "know" "now"
            normalized_window_str = " " + " ".join(window_tokens)

            for t in targets:
                # target alreadybut
                t_tokens = re.findall(r"[a-zA-Z0-9']+", t.lower())
                if not t_tokens:
                    continue

                normalized_target_str = " " + " ".join(t_tokens)

                # 2. afterafter target
                if normalized_window_str.endswith(normalized_target_str):
                    return t, True

            return None, False

        # pass via next_word
        matched_token_1, is_target_1 = is_valid_match(current_text_window, target_tokens_list_1, next_word)

        # when hallucination non_hallucination
        # "that that" first hallucination catch non-hal "that"
        matched_token_2 = None
        is_target_2 = False
        if not is_target_1:
            matched_token_2, is_target_2 = is_valid_match(current_text_window, target_tokens_list_2, next_word)

        if count < 60:
            if is_target_1 or is_target_2:
                print(
                    "DEBUG MATCH!",
                    "step=", count,
                    "is_hal=", is_target_1,
                    "is_non_hal=", is_target_2,
                    "matched1=", matched_token_1,
                    "matched2=", matched_token_2,
                    flush=True
                )

        if is_target_1 or is_target_2:
            print(
                "DEBUG ENTER ABLATION BLOCK",
                "step=", count,
                "matched_phrase=", (matched_token_1 if is_target_1 else matched_token_2),
                flush=True
            )
            # initmatrix[Layers x Heads]
            influences = [[None for _ in range(num_attention_heads)] for _ in range(num_hidden_layers)]

            # iterateand
            for layer_idx in range(len(o_proj_modules)):
                o_proj_module = o_proj_modules[layer_idx]
                for head_idx in range(num_attention_heads):
                    # Hook
                    hook_handle = o_proj_module.register_forward_pre_hook(attach_custom_hook(layer_idx, head_idx))
                    # =======================================================
                    # (fix)forcenone(use_cache=False)
                    # 1. model_inputs KV Cache ( 1296 vs 648 )
                    # 2. pipeline cache
                    # =======================================================
                    ablated_kwargs = model_kwargs.copy()
                    ablated_kwargs['use_cache'] = False
                    ablated_kwargs['past_key_values'] = None # cache
                    # cache_position cache
                    # ablated_kwargs["cache_position"] = torch.arange(input_ids.shape[1], device=input_ids.device)

                    # generate inputs (input_ids )
                    ablated_inputs = self.prepare_inputs_for_generation(input_ids, **ablated_kwargs)
                    ablated_inputs.pop("generation_config", None)

                    # (Ablated)
                    outputs_ablated = self(
                        # **model_inputs,
                        **ablated_inputs,
                        return_dict=True,
                        # output_attentions=True,
                        # output_hidden_states=True,
                        output_attentions=False,
                        output_hidden_states=False,
                        # use_cache=False,
                    )
                    next_token_logits_ablated = outputs_ablated.logits[:, -1, :]
                    ablated_probs = F.softmax(next_token_logits_ablated, dim=-1)
                    ablated_log_probs = F.log_softmax(next_token_logits_ablated, dim=-1)

                    #
                    token_id = next_tokens[0]
                    if influence_score_type == 'prob_diff':
                        val = (original_probs[0, token_id] - ablated_probs[0, token_id]).item()
                    elif influence_score_type == 'abs_prob_diff':
                        val = (original_probs[0, token_id] - ablated_probs[0, token_id]).abs().item()
                    elif influence_score_type == 'log_prob_diff':
                        val = (original_log_probs[0, token_id] - ablated_log_probs[0, token_id]).item()
                    else:
                        val = 0.0

                    influences[layer_idx][head_idx] = {
                        'layer': layer_idx,
                        'head': head_idx,
                        'original_prob': original_probs[0, token_id].item(),
                        'perturbed_prob': ablated_probs[0, token_id].item(),
                        'influence': val
                    }
                    hook_handle.remove()

            # Key matched_phrase="that that" -> Key="that that_25"
            matched_phrase = matched_token_1 if is_target_1 else matched_token_2
            key = f'{matched_phrase}_{count}'

            if is_target_1:
                hallucination_influences[key] = influences
            elif is_target_2:
                non_hallucination_influences[key] = influences

            torch.cuda.empty_cache()

        # ------------------ 5. continuegenerate ------------------

        if synced_gpus and this_peer_finished:
            continue

        count += 1

        if return_dict_in_generate:
            if output_scores: scores += (next_tokens_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,))
                if self.config.is_encoder_decoder: cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,))

        if eos_token_id is not None:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None: streamer.put(next_tokens.cpu())

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            if unfinished_sequences.max() == 0: this_peer_finished = True

        if stopping_criteria(input_ids, scores): this_peer_finished = True

        if this_peer_finished and not synced_gpus: break

    if streamer is not None: streamer.end()

    if return_dict_in_generate:
        # returnschema
        base_output = GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )
        return base_output, hallucination_influences, non_hallucination_influences
    else:
        return input_ids


def set_zero_ablation_greedy_search():
    # GM = transformers.generation.utils.GenerationMixin
    # GM.greedy_search = zero_ablation_greedy_search
    #
    # # transformers>=4.45 can generate call
    # if hasattr(GM, "_greedy_search"):
    #     GM._greedy_search = zero_ablation_greedy_search
    # transformers.generation.utils.GenerationMixin.greedy_search = zero_ablation_greedy_search

    GM = transformers.generation.utils.GenerationMixin

    # 1. override
    GM.greedy_search = zero_ablation_greedy_search

    # 2. overrideand sample sample path
    # 4.46 cancall _greedy_search _sample
    for method_name in ["_greedy_search", "sample", "_sample"]:
        if hasattr(GM, method_name):
            setattr(GM, method_name, zero_ablation_greedy_search)

    print(">>> Zero Ablation Patch applied to: greedy_search, _greedy_search, sample, _sample")