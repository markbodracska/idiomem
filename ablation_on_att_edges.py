import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import argparse
import transformer_lens
from transformer_lens import HookedTransformer

def merge_apostrophe_words(word_labels):
    merged_labels = []
    i = 0
    while i < len(word_labels):
        current_word = word_labels[i]['word']
        current_label = word_labels[i]['label']

        if i + 1 < len(word_labels) and word_labels[i + 1]['word'] in {"'s", "'re", "'ll", "'ve", "'d", "n't"}:
            # Merge the next token into the current one
            next_word = word_labels[i + 1]['word']
            merged_word = current_word + next_word
            # Use the same label or choose 'KEYWORD' if either is keyword
            merged_label = 'KEYWORD' if current_label == 'KEYWORD' or word_labels[i + 1]['label'] == 'KEYWORD' else 'NON-KEYWORD'
            merged_labels.append({'word': merged_word, 'label': merged_label})
            i += 2
        else:
            merged_labels.append({'word': current_word, 'label': current_label})
            i += 1

    return merged_labels

def tokenize_without_bos(text, model):
    tokens = model.to_tokens(text)[0]
    # If the first token is BOS, remove it for token count purposes
    return tokens[1:] if tokens[0] == model.tokenizer.bos_token_id else tokens

def get_labeled_token_positions(idiom_df, idiom, tokenized_prompt, model, label_type, dataset):
    """
    Extract token positions for a specified label type ('KEYWORD' or 'NON-KEYWORD') from a tokenized prompt.

    Args:
        idiom_df: DataFrame with 'idiom' and 'labeled_words' columns.
        idiom: The idiom string to look up in the DataFrame.
        tokenized_prompt: The tokenized prompt (tensor).
        model: The transformer model with `to_tokens()` method.
        label_type: 'KEYWORD' or 'NON-KEYWORD'.
        dataset: Either 'idiomem' (no context) or 'magpie' (with context).

    Returns:
        A list of token positions matching the label type.
    """
    raw_labels = idiom_df.loc[idiom_df['idiom'] == idiom, 'labeled_words'].values[0]
    merged_labels = merge_apostrophe_words(raw_labels)
    word_labels = merged_labels[:-1]  # Drop final entry if it's a sentinel
    word_list = [entry['word'] for entry in word_labels]

    if dataset == "idiomem":
        # Determine starting token index (skip BOS if present)
        full_idiom_text = " ".join(word_list)
        idiom_tokens = model.to_tokens(full_idiom_text)
        #print(tokenized_prompt)
        #print(idiom_tokens)
        start_index = 1 if len(tokenized_prompt) == len(idiom_tokens[0]) else 0
        #print(len(tokenized_prompt))
        #print(len(idiom_tokens[0]))

        token_positions = []
        token_pointer = start_index
        for i, word in enumerate(word_list):
            # Tokenize word with leading space when needed
            word_text = " " + word if i > 0 else word
            tokens = model.to_tokens(word_text)[0, 1:]
            
            token_indices = list(range(token_pointer, token_pointer + len(tokens)))

            # Skip BOS *from labeling only*, not from counting
            if tokens[0] == model.tokenizer.bos_token_id:
                token_indices_to_label = token_indices[1:]
            else:
                token_indices_to_label = token_indices

            if word_labels[i]['label'] == label_type:
                token_positions.extend(token_indices_to_label)

            token_pointer += len(tokens)  # Always count all tokens to stay aligned
        return token_positions

    elif dataset == "magpie":
        # Tokenize idiom to locate it inside the full prompt
        idiom_tokens = []
        for word in word_list:
            tokens = model.to_tokens(" " + word)[0, 1:]
            idiom_tokens.extend(tokens.tolist())

        prompt_tokens = tokenized_prompt.tolist()
        for start_index in range(len(prompt_tokens) - len(idiom_tokens) + 1):
            if prompt_tokens[start_index:start_index + len(idiom_tokens)] == idiom_tokens:
                idiom_start = start_index
                break
        else:
            raise ValueError("Idiom tokens not found in prompt")

        token_positions = []
        pointer = idiom_start
        for i, word in enumerate(word_list):
            tokens = model.to_tokens(" " + word)[0, 1:]
            if word_labels[i]['label'] == label_type:
                token_positions.extend(range(pointer, pointer + len(tokens)))
            pointer += len(tokens)
        return token_positions

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

# Function to ablate specific attention edges
def ablate_attention_edges(attn_scores, hook, keyword_mask):
    attn_scores[:, :, -1, keyword_mask] = torch.finfo(attn_scores.dtype).min
    return attn_scores

def load_data_and_model(args):
    # Load model
    model = HookedTransformer.from_pretrained(args.model_name)
    model.eval()

    # Load idioms and labels
    idioms_labeled_df = pd.read_json(args.idioms_path, lines=True)
    idioms = idioms_labeled_df['idiom'].tolist()

    return model, idioms, idioms_labeled_df

# one-by-one ablation
def ablate_idiom(model, idiom, idioms_labeled_df, word_type, dataset_name, num_layers, device, window_size, skip_bos):
    individual_results = []
    
    prefix = idiom.split()[:-1]
    prompt = ' '.join(prefix)
    true_last_word = ' ' + idiom.split()[-1]

    tokens = model.to_tokens(prompt).to(device)[0, 1:] if skip_bos else model.to_tokens(prompt).to(device)[0, :]
    true_tokenized_ids = model.to_tokens(true_last_word).to(device)[0, 1:]
    full_word_token_id = true_tokenized_ids[0]

    with torch.no_grad():
        original_logits = model(tokens)[0, -1]
        predicted_id = torch.argmax(original_logits).item()
        original_prob = F.softmax(original_logits, dim=-1)[predicted_id].item()
        original_ranks = torch.argsort(original_logits, descending=True)

    correct_prediction = predicted_id == full_word_token_id

    token_positions = get_labeled_token_positions(idioms_labeled_df, idiom, tokens, model, word_type, dataset_name)

    for pos in token_positions:
        for layer in range(num_layers):
            start_layer = max(0, layer - window_size // 2)
            end_layer = min(num_layers, layer + window_size // 2 + 1)

            def make_hook(position):
                return lambda attn_scores, hook: ablate_attention_edges(attn_scores, hook, [position])

            for abl_layer in range(start_layer, end_layer):
                hook_name = f"blocks.{abl_layer}.attn.hook_attn_scores"
                model.add_hook(hook_name, make_hook(pos))

            with torch.no_grad():
                ablated_logits = model(tokens)[0, -1]
                ablated_probs = F.softmax(ablated_logits, dim=-1)
                ablate_predicted_id = torch.argmax(ablated_logits).item()
                ablated_prob = ablated_probs[predicted_id].item()
                prob_abs_diff = ablated_prob - original_prob
                prob_rel_diff = prob_abs_diff / original_prob
                ablated_token_original_rank = (original_ranks == ablate_predicted_id).nonzero(as_tuple=True)[0].item()
                topk_ids = torch.topk(ablated_logits, 10).indices.tolist()
                topk_tokens = [model.to_string(tok_id) for tok_id in topk_ids]

            individual_results.append({
                "idiom": idiom,
                "layer": layer,
                "ablation_position": pos,
                "predicted_token": model.to_string(predicted_id),
                "ablated_predicted_token": model.to_string(ablate_predicted_id),
                "correct_token": model.to_string(full_word_token_id),
                "correct_prediction": correct_prediction,
                "original_prob": original_prob,
                "ablated_prob": ablated_prob,
                "abs_diff": prob_abs_diff,
                "rel_diff": prob_rel_diff,
                "ablated_token_original_rank": ablated_token_original_rank,
                "top10_after_ablation": ", ".join(topk_tokens)
            })

            
            model.reset_hooks()
    
    return individual_results

def main():
    parser = argparse.ArgumentParser(description="Run keyword edge ablation on idioms without context.")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--idioms_path", type=str, required=True, help="CSV file with idioms with labeled positions")
    parser.add_argument("--output_path", type=str, help="Path to save the results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument("--word_type", type=str, choices=["KEYWORD", "NON-KEYWORD"], default="KEYWORD", help="Type of token to ablate")
    parser.add_argument("--dataset", type=str, default="idiomem", help="Dataset name used for position lookup")
    parser.add_argument("--skip_bos", action="store_true", help="Skip the BOS token in tokenized inputs")
    parser.add_argument("--window_sizes", type=int, nargs='+', default=[5], help="List of layer window sizes for edge ablation")
    args = parser.parse_args()

    safe_model_name = args.model_name.replace("/", "_")

    model, idioms, idioms_labeled_df = load_data_and_model(args)
    num_layers = model.cfg.n_layers

    all_individual_results = []
    
    for word_type in ["KEYWORD", "NON-KEYWORD"]:
        for window_size in args.window_sizes:
            print(f"Running ablation for word_type={word_type}, window_size={window_size}")
            for idiom in tqdm(idioms, desc=f"{word_type} | window {window_size}"):
                results = ablate_idiom(model, idiom, idioms_labeled_df, word_type, args.dataset, num_layers, args.device, window_size, args.skip_bos)
                all_individual_results.extend(results)
                
            suffix = f"{safe_model_name}_ds-{args.dataset}_wt-{word_type}_ws-{window_size}" + ("_skipbos" if args.skip_bos else "")
            output_path = f"{args.output_path}/{safe_model_name}/{suffix}.csv"
            individual_results_df = pd.DataFrame(all_individual_results)
            individual_results_df.to_csv(output_path, index=False)

            # Reset for next run
            all_individual_results.clear()
        

if __name__ == "__main__":
    main()
