import argparse
import os
import numpy as np
import pandas as pd
import torch

from lm_debugger_utils import get_examples_df_for_prompts_gpt2, get_examples_df_for_prompts_pythia
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForMaskedLM, GPTNeoXForCausalLM
from tqdm import tqdm


def parse_args():
    def nullable_string(val):
        if val is None or val.lower() == 'none':
            return None
        return val

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_df_path",
                        default=r"data/gpt2-medium_correct_idioms.pkl",
                        type=nullable_string,
                        required=False)

    parser.add_argument("--output_df_path",
                        type=str,
                        required=True)

    parser.add_argument("--sample_size",
                        default=30,
                        type=int,
                        help="how many instances to sample, -1 for using all instances.",
                        required=False)

    parser.add_argument("--model",
                        default=r"gpt2-medium",
                        type=str,
                        required=False)

    parser.add_argument("--pred_as_target",
                        default=False,
                        action='store_true',
                        required=False)

    return parser.parse_args()


def main(model, model_type, tokenizer, input_df, sample_size, output_df_path, pred_as_target):
    if model_type == "gpt2":
        num_layers = model.config.n_layer
    else: # pythia
        num_layers = model.config.num_hidden_layers

    if isinstance(input_df, str):
        df = pd.read_pickle(input_df)
        print(f"[-] loaded instances from: {input_df}")
    else:
        df = input_df
    if sample_size > 0:
        effective_sample_size = min(sample_size, len(df))
        df = df.sample(n=effective_sample_size)
        print(f"[-] sampled {effective_sample_size} instances.")
    dict_tmp = df[["prompt", f"top_coef_idx"]].to_dict(orient="list")
    prompt_to_top_coef = dict(zip(dict_tmp["prompt"], dict_tmp[f"top_coef_idx"]))

    # create knockout configurations
    print("[-] creating knockout configurations...")
    configs = {}
    #for l in range(model.config.n_layer):
    for l in range(num_layers):

        # knockouts of top value vectors in consecutive layers.
        for r in [1, 2, 3, 4, 5]:
            for k in [1, 10, 100, 1000]:
                #if l <= model.config.n_layer - r:
                if l <= num_layers - r:
                    prompt_to_rl_all_top_coef_k = {
                        prompt: {l + ri: top_coef[l + ri][:k] for ri in range(r)} for prompt, top_coef in
                        prompt_to_top_coef.items()}
                    configs[f"layers{l}-{l + r - 1}_top{k}"] = prompt_to_rl_all_top_coef_k

        # knockouts of non-top value vectors in consecutive layers.
        if model_type == "gpt2":
            hidden_dim = model.transformer.h[0].mlp.c_proj.weight.size(0)
        else: # pythia
            hidden_dim = model.gpt_neox.layers[0].mlp.dense_4h_to_h.weight.size(1)
            
        all_dims = np.arange(hidden_dim)
        for r in [1, 2, 3]:
            for k in [10, 100, 1000]:
                #if l <= model.config.n_layer - r:
                if l <= num_layers - r:
                    prompt_to_rl_all_nontop_coef_k = {
                        prompt: {
                            l + ri: np.setdiff1d(all_dims, top_coef[l + ri][:k]).tolist()
                            for ri in range(r)
                        }
                        for prompt, top_coef in prompt_to_top_coef.items()
                    }
                    configs[f"layers{l}-{l + r - 1}_nontop{k}"] = prompt_to_rl_all_nontop_coef_k

    if pred_as_target:
        dict_tmp = df[["prompt", "pred", "pred_token"]].to_dict(orient="list")
        targets = dict(zip(dict_tmp["prompt"], dict_tmp["pred"]))
        target_tokens = dict(zip(dict_tmp["prompt"], dict_tmp["pred_token"]))
        full_targets = targets
    else:
        dict_tmp = df[["prompt", "target", "target_token", "full_target"]].to_dict(orient="list")
        targets = dict(zip(dict_tmp["prompt"], dict_tmp["target"]))
        target_tokens = dict(zip(dict_tmp["prompt"], dict_tmp["target_token"]))
        full_targets = dict(zip(dict_tmp["prompt"], dict_tmp["full_target"]))

    # targets_info = {"targets": targets, "target_tokens": target_tokens}
    targets_info = {"targets": targets, "target_tokens": target_tokens, "full_targets": full_targets}

    # execute knockouts
    print("[-] executing knockouts!")
    results = []
    for config_name, config in tqdm(configs.items()):
        prompts = list(config.keys())
        if model_type == "gpt2":
            dfko = get_examples_df_for_prompts_gpt2(prompts, targets_info, model, tokenizer,
                                           top_k=10, knockout_config=config)
        else: # pythia
            dfko = get_examples_df_for_prompts_pythia(prompts, targets_info, model, tokenizer,
                                           top_k=10, knockout_config=config)

        #dfko = get_examples_df_for_prompts_gpt2(prompts, targets_info, model, tokenizer,
        #                                   top_k=10, knockout_config=config)
        dfko["config"] = config_name
        results.append(dfko)

    # store the merged df with all knockout results
    print(f"[-] merging results from {len(configs)} knockout experiments...")
    df = df[[col for col in df.columns if "top_coef_idx_" not in col]]
    df_results = pd.concat([df] + results).reset_index()
    df_results.to_pickle(output_df_path)
    print(f"[-] wrote results to: {output_df_path}")
    return output_df_path


if __name__ == '__main__':
    args = parse_args()

    model_name = args.model.strip()
    if 'gpt2' in model_name:
        model_ = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer_ = GPT2Tokenizer.from_pretrained(model_name)
        num_layers = model_.config.n_layer
        model_type = 'gpt2'
    else: # pythia models
        model_ = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/{model_name}")
        tokenizer_ = AutoTokenizer.from_pretrained(f"EleutherAI/{model_name}")
        num_layers = model_.config.num_hidden_layers
        model_type = 'pythia'

    if torch.cuda.is_available():
        model_.cuda()
    model_.eval()

    main(model_, tokenizer_,
         args.input_df_path, args.sample_size, args.output_df_path,
         args.pred_as_target)
