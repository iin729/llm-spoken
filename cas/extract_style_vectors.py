# extract_style_vector_torch.py
import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Prompt template (unchanged, ensures model understands the spec)
SYSTEM_PROMPT_TEMPLATE = """### Instruction
You are an expert conversationalist. Your task is to convert formal written language into spontaneous spoken language at a specified **Orality Level (0-5)** with **Semantic Preservation**. Please directly output the rewritten spoken text without any explanations or formatting symbols.

### Orality Level Definitions
You must adjust the density of orality words and phrases, repetition, and sentence fragmentation based on the target level:

- Level 0 (written): Formal structure. High lexical density. No fillers. No extra interaction. No repetition.
- Level 1 (spoken, slight): Light contraction and mild simplification. Rare discourse markers. No repetition.
- Level 2 (spoken, low): Add occasional fillers (um/uh/okay) and light turn-organization (well/so). Slight sentence splitting.
- Level 3 (spoken, medium): Clear conversational flow. Some stance/hedges (I think/maybe). More fragmentation. Small, natural repetitions allowed.
- Level 4 (spoken, high): Frequent interaction cues (you know/right?). More hedging. Noticeable repetition and shorter clauses, but still coherent.
- Level 5 (spoken, very high): Strongly spontaneous. Frequent fillers + discourse markers + turn-yielding cues. More repetition and repairs. Still preserve every fact.

### Semantic Preservation

- Preserve Core Information: Do not add, invent, change, or omit substantive content (e.g., entities, numbers, conditions, stances).
- Maintain Logical Integrity: Ensure the core meaning and intent are clear. While sentence structure should be simplified, the original logical order and core meaning must be maintained.

### Let's begin
Please directly output the one spoken text at a specified Orality Level.
"""


def build_prompt_messages(written_text, target_level):
    user_content = f"**Written Language Level 0:**\n{written_text}\n\n**Spoken Language Level {target_level}:**\n"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": user_content}
    ]
    return messages


class ActivationCapturer:
    """Lightweight class for capturing intermediate-layer activations"""

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []

    def register_hooks(self):
        """register hooks on every layer"""
        # Qwen2.5 layer path: model.model.layers
        layers = self.model.model.layers
        print(f"[DEBUG] Registering hooks for {len(layers)} layers...")

        for idx, layer in enumerate(layers):
            # default-arg trick to capture loop index
            def hook_fn(module, input, output, idx=idx):
                # output is a tuple (hidden_states, ...)
                # we only need hidden_states
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output

                # take last-token hidden state: [Batch, Seq, Dim] -> [Batch, Dim]
                # move to CPU immediately to save VRAM
                self.activations[idx] = hidden[:, -1, :].detach().cpu()

            # register forward hook
            handle = layer.register_forward_hook(hook_fn)
            self.hooks.append(handle)

    def remove_hooks(self):
        """remove hooks and free resources"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def clear_activations(self):
        self.activations = {}


def main(args):
    print(f"Loading model: {args.model_path}")
    print(f"Mode: Pure PyTorch Hooks | Level {args.target_level} - Level {args.base_level}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
        # attn_implementation="flash_attention_2"  # enable on A100/H100/etc
    )
    model.eval()  # eval mode

    # init Hook catch
    capturer = ActivationCapturer(model)
    capturer.register_hooks()

    print(f"Loading data from: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if args.num_samples > 0:
        data = data[:args.num_samples]

    print(f"Processing {len(data)} samples...")

    diff_sums = {}
    count = 0
    batch_size = args.batch_size

    # disable
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch_data = data[i: i + batch_size]

            # 1.
            texts_target = []
            texts_base = []
            for item in batch_data:
                # Target
                msgs = build_prompt_messages(item['written'], target_level=args.target_level)
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                texts_target.append(text)
                # Base
                msgs = build_prompt_messages(item['written'], target_level=args.base_level)
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                texts_base.append(text)

            # 2. Tokenize
            inputs_target = tokenizer(texts_target, return_tensors="pt", padding=True).to(model.device)
            inputs_base = tokenizer(texts_base, return_tensors="pt", padding=True).to(model.device)

            try:
                # 3. - Target
                capturer.clear_activations()
                model(**inputs_target)
                # at this point capturer.activations already Target vector ( CPU )
                target_acts = capturer.activations.copy() # dict

                # 4. - Base
                capturer.clear_activations()
                model(**inputs_base)
                base_acts = capturer.activations.copy()

                # 5. ( CPU row)
                current_bs = len(batch_data)
                count += current_bs

                for layer_idx in target_acts:
                    vec_target = target_acts[layer_idx].float()  # CPU tensor
                    vec_base = base_acts[layer_idx].float()  # CPU tensor

                    diff = vec_target - vec_base
                    mean_diff = diff.sum(dim=0)

                    if layer_idx not in diff_sums:
                        diff_sums[layer_idx] = mean_diff
                    else:
                        diff_sums[layer_idx] += mean_diff

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"[ERROR] OOM in batch {i}. Skipping.")
                    torch.cuda.empty_cache()
                else:
                    print(f"[ERROR] Batch {i} failed: {e}")
                continue

            # cleanup
            del inputs_target, inputs_base
            torch.cuda.empty_cache()

    # remove Hooks
    capturer.remove_hooks()

    # save
    final_vectors = {}
    print(f"\nComputing mean vectors over {count} samples...")

    if len(diff_sums) == 0:
        print("[CRITICAL ERROR] No vectors extracted.")
    else:
        for layer_idx, sum_tensor in diff_sums.items():
            mean_vector = sum_tensor / count
            final_vectors[layer_idx] = mean_vector.to(torch.bfloat16)

        os.makedirs(args.output_path, exist_ok=True)
        filename = f"style_vectors_l{args.target_level}_minus_l{args.base_level}.pt"
        save_file = os.path.join(args.output_path, filename)
        torch.save(final_vectors, save_file)
        print(f"Saved style vectors to: {save_file}")
        print(f"Vector keys (layers): {list(final_vectors.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4) # PyTorch Batch=4
    parser.add_argument("--target_level", type=int, default=5)
    parser.add_argument("--base_level", type=int, default=0)

    args = parser.parse_args()
    main(args)