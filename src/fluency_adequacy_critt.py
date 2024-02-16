import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM,  AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, \
    LlamaTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
import numpy as np
import os
from os.path import basename

import warnings
warnings.filterwarnings("ignore")

from data import SingleModelCRITTDataset
from model import load_model_tokenizer, get_log_probs

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default=None, type=str, required=True,
                        help="data source directory")
    parser.add_argument("--out_dir", default=None, type=str, required=True,
                        help="output directory")
    parser.add_argument("--num_workers", default=1, type=int,
                        help="number of workers")
    parser.add_argument("--m2m100", action="store_true", help="Use M2M100 MT model for log prob")
    parser.add_argument("--cpu", action='store_true', help="cpu instead of cuda")
    args = parser.parse_args()

    fpath = os.path.join(args.source_dir, 'critt-sentences.tsv')

    if args.m2m100:
        mt_model, mt_tokenizer = load_model_tokenizer(M2M100ForConditionalGeneration, M2M100Tokenizer,
                                                      "facebook/m2m100_1.2B")
    else:
        mt_model, mt_tokenizer = load_model_tokenizer(AutoModelForSeq2SeqLM, AutoTokenizer, "facebook/nllb-200-3.3B",
                                                      pad_token=None, pad_token_id=None)

    mt_xentropy = nn.CrossEntropyLoss(ignore_index=mt_tokenizer.pad_token_id, reduction='none')


    device = torch.device("cpu")
    if not args.cpu and torch.cuda.is_available():
        mt_model.cuda()
        device = torch.device("cuda")
    mt_model.eval()

    use_lang_code = False if args.m2m100 else True
    dataset = SingleModelCRITTDataset(fpath, mt_tokenizer, mt_model=True, input_split_into_words=True,
                                      use_lang_code=use_lang_code)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                            collate_fn=dataset.tokenize)  # sentences have different language pair, process line by line
    column_names = ["mt_forward_lprobs_mean", "mt_backward_lprobs_mean", "mt_forward_lprobs_sum",
                    "mt_backward_lprobs_sum", "mt_forward_lprobs_min", "mt_backward_lprobs_min"]


    skipped_dataset = SingleModelCRITTDataset(fpath, mt_tokenizer, mt_model=True, input_split_into_words=True,
                                              skip_src=True, use_lang_code=use_lang_code)
    skipped_dataloader = DataLoader(skipped_dataset, batch_size=1, shuffle=False, drop_last=False,
                                    collate_fn=skipped_dataset.tokenize)
    skipped_column_names = ["skipped_forward_lprobs_mean", "skipped_backward_lprobs_mean", "skipped_forward_lprobs_sum",
                            "skipped_backward_lprobs_sum", "skipped_forward_lprobs_min", "skipped_backward_lprobs_min"]


    with torch.no_grad():


        for dloader, col_names in zip([dataloader,  skipped_dataloader], [column_names, skipped_column_names]):
            mt_forward_lprobs_mean, mt_backward_lprobs_mean, mt_forward_lprobs_sum, mt_backward_lprobs_sum = [], [], [], []
            mt_forward_lprobs_min, mt_backward_lprobs_min = [], []
            for bid, batch in enumerate(dloader):
                mt_forward_inputs, mt_backward_inputs = batch
                mt_fd_lp_mean, mt_fd_lp_sum, mt_fd_lp_min = get_log_probs(mt_forward_inputs, mt_model, device, mt_xentropy,
                                                            mt_forward_inputs.labels)
                mt_bd_lp_mean, mt_bd_lp_sum, mt_bd_lp_min = get_log_probs(mt_backward_inputs, mt_model, device, mt_xentropy,
                                                            mt_backward_inputs.labels)

                mt_forward_lprobs_mean += mt_fd_lp_mean
                mt_forward_lprobs_sum += mt_fd_lp_sum
                mt_backward_lprobs_mean += mt_bd_lp_mean
                mt_backward_lprobs_sum += mt_bd_lp_sum
                mt_forward_lprobs_min += mt_fd_lp_min
                mt_backward_lprobs_min += mt_bd_lp_min

            values = [mt_forward_lprobs_mean, mt_backward_lprobs_mean, mt_forward_lprobs_sum, mt_backward_lprobs_sum,
                      mt_forward_lprobs_min, mt_backward_lprobs_min]
            for mname, measure in zip(col_names, values):
                dataset.data[mname] = np.array(measure)

    data = dataset.data
    data.to_csv(os.path.join(args.out_dir, basename(fpath)), sep='\t', index=False)

    del mt_model, mt_xentropy, mt_tokenizer

    # LM estimator ##############################
    lm_model, lm_tokenizer = load_model_tokenizer(AutoModelForCausalLM, LlamaTokenizer, "meta-llama/Llama-2-7b-hf",
                                                  tokenizer_padding_side='left')
    lm_tokenizer.pad_token = lm_tokenizer.bos_token
    lm_tokenizer.pad_token_id = lm_tokenizer.bos_token_id
    lm_model.config.pad_token_id = lm_model.config.bos_token_id
    lm_model, lm_tokenizer = load_model_tokenizer(AutoModelForCausalLM, AutoTokenizer, "bigscience/bloom-7b1",
                                                    tokenizer_padding_side='left', add_prefix_space=True)

    lm_xentropy = nn.CrossEntropyLoss(ignore_index=lm_tokenizer.pad_token_id, reduction='none')

    device = torch.device("cpu")
    if not args.cpu and torch.cuda.is_available():
        lm_model.cuda()
        device = torch.device("cuda")

    lm_model.eval()
    lm_src_lprobs_mean, lm_tgt_lprobs_mean, lm_src_lprobs_sum, lm_tgt_lprobs_sum, lm_src_lprobs_min, lm_tgt_lprobs_min = [], [], [], [], [], []

    with torch.no_grad():
        dataset = SingleModelCRITTDataset(fpath, lm_tokenizer, mt_model=False, input_split_into_words=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                                collate_fn=dataset.tokenize)  # sentences have different language pair, process line by line

        for bid, batch in enumerate(dataloader):
            lm_src_inputs, lm_tgt_inputs = batch

            lm_src_lp_mean, lm_src_lp_sum, lm_src_lp_min = get_log_probs(lm_src_inputs, lm_model, device, lm_xentropy,
                                                          lm_src_inputs.input_ids)
            lm_tgt_lp_mean, lm_tgt_lp_sum, lm_tgt_lp_min  = get_log_probs(lm_tgt_inputs, lm_model, device, lm_xentropy,
                                                          lm_tgt_inputs.input_ids)

            lm_src_lprobs_mean += lm_src_lp_mean
            lm_src_lprobs_sum += lm_src_lp_sum
            lm_tgt_lprobs_mean += lm_tgt_lp_mean
            lm_tgt_lprobs_sum += lm_tgt_lp_sum
            lm_src_lprobs_min += lm_src_lp_min
            lm_tgt_lprobs_min += lm_tgt_lp_min

    for mname, measure in {"lm_src_lprobs_mean": lm_src_lprobs_mean, "lm_tgt_lprobs_mean": lm_tgt_lprobs_mean,
                           "lm_src_lprobs_sum": lm_src_lprobs_sum, "lm_tgt_lprobs_sum": lm_tgt_lprobs_sum,
                           "lm_src_lprobs_min": lm_src_lprobs_min, "lm_tgt_lprobs_min": lm_tgt_lprobs_min}.items():
        dataset.data[mname] = np.array(measure)
    dataset.data.to_csv(os.path.join(args.out_dir, basename(fpath)), sep='\t', index=False)

if __name__ == "__main__":
    main()



