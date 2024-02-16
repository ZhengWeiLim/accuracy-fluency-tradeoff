import torch
from transformers import AutoModelForCausalLM

def load_model_tokenizer(model_class, tokenizer_class, model_checkpoint, pad_token=None, pad_token_id=None,
                         tokenizer_padding_side=None,  add_prefix_space=False):
    if isinstance(model_class, AutoModelForCausalLM):
        model = model_class.from_pretrained(model_checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
    else:
        model = model_class.from_pretrained(model_checkpoint, torch_dtype=torch.bfloat16)

    if tokenizer_padding_side is not None:
        tokenizer = tokenizer_class.from_pretrained(model_checkpoint, padding_side=tokenizer_padding_side, add_prefix_space=add_prefix_space)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_checkpoint, add_prefix_space=add_prefix_space)

    if pad_token is not None:
        tokenizer.pad_token = pad_token
    if pad_token_id is not None:
        tokenizer.pad_token_id = pad_token_id


    return model, tokenizer


def get_log_probs(inputs, model, device, xentropy, labels, loss_mask=None):
    if device is not None:
        inputs, labels = inputs.to(device), labels.to(device)
    output = model(**inputs)

    xent = xentropy(output.logits.permute(0,2,1), labels)
    if loss_mask is not None:
        if device is not None:
            loss_mask = loss_mask.to(device)
        xent *= loss_mask

    ntokens = (xent!=0).sum(dim=-1)
    lprobs_min = - torch.max(xent, dim=-1).values
    lprobs_sum = (-torch.sum(xent, dim=-1))
    lprobs_mean = lprobs_sum / ntokens

    return lprobs_mean.tolist(), lprobs_sum.tolist(), lprobs_min.tolist()
