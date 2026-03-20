import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_gsm8k(model, tokenizer, num_samples=200):
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(num_samples))
    
    correct = 0

    for item in dataset:
        # Format prompt
        prompt = f"Question: {item['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate answer (greedy decoding — simple and reproducible)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False        # greedy
            )
        
        # Decode only the newly generated tokens
        generated = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract the ground truth number from GSM8K
        # Ground truth answers end with #### <number>
        ground_truth = item['answer'].split('####')[-1].strip()
        
        # Check if correct number appears in generated text
        if ground_truth in generated:
            correct += 1
    
    accuracy = correct / num_samples
    print(f"GSM8K Accuracy: {accuracy:.3f} ({correct}/{num_samples})")
    return accuracy

def score_choice(model, tokenizer, prompt, choice):
    """Compute log-likelihood of a choice given a prompt."""
    full_text = prompt + " " + choice
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    prompt_len = prompt_inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        # Get per-token log probs
        logits = outputs.logits[:, :-1, :]
        labels = inputs['input_ids'][:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Only score the choice tokens, not the prompt tokens
        choice_log_probs = log_probs[:, prompt_len-1:, :]
        choice_labels = labels[:, prompt_len-1:]
        score = choice_log_probs.gather(
            2, choice_labels.unsqueeze(-1)
        ).squeeze(-1).mean()
    
    return score.item()

def evaluate_arc(model, tokenizer, num_samples=200):
    dataset = load_dataset(
        "ai2_arc", "ARC-Challenge", split="test"
    )
    dataset = dataset.select(range(num_samples))
    
    correct = 0

    for item in dataset:
        prompt = f"Question: {item['question']}\nAnswer:"
        choices = item['choices']['text']
        labels = item['choices']['label']  # ['A', 'B', 'C', 'D']
        
        # Score each choice
        scores = [
            score_choice(model, tokenizer, prompt, choice)
            for choice in choices
        ]
        
        # Pick highest scoring choice
        predicted_label = labels[np.argmax(scores)]
        
        if predicted_label == item['answerKey']:
            correct += 1
    
    accuracy = correct / num_samples
    print(f"ARC-Challenge Accuracy: {accuracy:.3f} ({correct}/{num_samples})")
    return accuracy

def evaluate_perplexity(model, tokenizer, num_samples=50):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Concatenate text samples and chunk into fixed-length windows
    text = "\n\n".join(dataset['text'][:num_samples])
    encodings = tokenizer(text, return_tensors="pt")
    
    max_length = 512          # process 512 tokens at a time
    stride = 256              # overlap windows to avoid edge effects
    seq_len = encodings.input_ids.shape[1]
    
    nlls = []                 # negative log likelihoods

    for begin in range(0, seq_len - max_length, stride):
        end = begin + max_length
        input_ids = encodings.input_ids[:, begin:end].to(device)
        
        # Only compute loss on the non-overlapping part
        # (to avoid counting tokens twice)
        target_len = min(stride, max_length)
        target_ids = input_ids.clone()
        target_ids[:, :-target_len] = -100  # -100 = ignore in loss
        
        with torch.no_grad():
            loss = model(
                input_ids, labels=target_ids
            ).loss
        
        nlls.append(loss.item())
    
    perplexity = np.exp(np.mean(nlls))
    print(f"Perplexity (WikiText-2): {perplexity:.2f}")
    return perplexity