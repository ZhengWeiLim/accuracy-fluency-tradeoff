import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, \
    LlamaTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
import numpy as np
import os, glob
from os.path import basename
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

from data import SingleModelTranslationDataset, TranslationeseMTDataset, lang_code
from model import load_model_tokenizer, get_log_probs

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default=None, type=str,
                        help="data source directory with all <language pair>.tsv files")
    parser.add_argument("--source_file", default=None, type=str,
                        help="data source file (tsv)")
    parser.add_argument("--src_lang", default=None, type=str,
                        help="source langauge")
    parser.add_argument("--tgt_lang", default=None, type=str,
                        help="target langauge")
    parser.add_argument("--out_dir", default=None, type=str, required=True,
                        help="output directory")
    parser.add_argument("--bsz", default=16, type=int,
                        help="batch size")
    parser.add_argument("--csv_quoting", default=3, type=int,
                        help="csv quoting when reading file")
    parser.add_argument("--bloom", action="store_true", help="Use LLM Bloom for log prob")
    parser.add_argument("--nllb", action="store_true", help="Use NLLB MT model for log prob")
    parser.add_argument("--m2m100", action="store_true", help="Use M2M100 MT model for log prob")
    parser.add_argument("--cpu", action='store_true', help="cpu instead of cuda")
    parser.add_argument("--mask_src", action='store_true', help='mask input in MT model')
    parser.add_argument("--skip_src", action='store_true', help='skip input in MT model')
    args = parser.parse_args()
    fname_df = {}
    device = torch.device("cpu")
    fpaths = glob.glob(os.path.join(args.source_dir, "*.tsv")) if args.source_dir is not None else [args.source_file]

    # LM estimator
    if args.bloom:
        llm_model, llm_tokenizer = load_model_tokenizer(AutoModelForCausalLM, AutoTokenizer, "bigscience/bloom-7b1", tokenizer_padding_side='left')

        llm_model.eval()
        xentropy = nn.CrossEntropyLoss(ignore_index=llm_tokenizer.pad_token_id, reduction='none')

        if not args.cpu and torch.cuda.is_available():
            llm_model.cuda()
            device = torch.device("cuda")

        for fpath in fpaths:

            lm_src_lprobs_mean, lm_tgt_lprobs_mean, lm_src_lprobs_sum, lm_tgt_lprobs_sum, lm_src_lprobs_min, lm_tgt_lprobs_min = [], [], [], [], [], []

            with torch.no_grad():
                dataset = SingleModelTranslationDataset(fpath, llm_tokenizer, csv_quoting=args.csv_quoting)
                dataloader = DataLoader(dataset, batch_size=args.bsz, shuffle=False, drop_last=False,
                                        collate_fn=dataset.tokenize)

                for bid, batch in enumerate(dataloader):
                    lm_src_inputs, lm_tgt_inputs = batch

                    lm_src_lp_mean, lm_src_lp_sum, lm_src_lp_min = get_log_probs(lm_src_inputs, llm_model, device, xentropy,
                                                                  lm_src_inputs.input_ids)
                    lm_tgt_lp_mean, lm_tgt_lp_sum, lm_tgt_lp_min = get_log_probs(lm_tgt_inputs, llm_model, device, xentropy,
                                                                  lm_tgt_inputs.input_ids)

                    lm_src_lprobs_mean += lm_src_lp_mean
                    lm_src_lprobs_sum += lm_src_lp_sum
                    lm_src_lprobs_min += lm_src_lp_min
                    lm_tgt_lprobs_mean += lm_tgt_lp_mean
                    lm_tgt_lprobs_sum += lm_tgt_lp_sum
                    lm_tgt_lprobs_min += lm_tgt_lp_min



                for mname, measure in {"lm_src_lprobs_mean": lm_src_lprobs_mean, "lm_tgt_lprobs_mean": lm_tgt_lprobs_mean,
                                       "lm_src_lprobs_sum": lm_src_lprobs_sum, "lm_tgt_lprobs_sum": lm_tgt_lprobs_sum,
                                       "lm_src_lprobs_min": lm_src_lprobs_min, "lm_tgt_lprobs_min": lm_tgt_lprobs_min,
                                       }.items():
                    dataset.data[mname] = np.array(measure)

                dataset.data.to_csv(os.path.join(args.out_dir, basename(fpath)), sep='\t', index=False)

                fname_df[basename(fpath)] = deepcopy(dataset.data)



        del llm_tokenizer
        del llm_model
        del dataset
        del dataloader



    if args.nllb or args.m2m100:
        # MT estimator

        if args.nllb:
            mt_model, mt_tokenizer = load_model_tokenizer(AutoModelForSeq2SeqLM, AutoTokenizer, "facebook/nllb-200-3.3B")
        else:
            mt_model, mt_tokenizer = load_model_tokenizer(M2M100ForConditionalGeneration, M2M100Tokenizer,
                                                          "facebook/m2m100_1.2B")
        mt_model.eval()

        xentropy = nn.CrossEntropyLoss(ignore_index=mt_tokenizer.pad_token_id, reduction='none')

        if not args.cpu and torch.cuda.is_available():
            mt_model.cuda()
            device = torch.device("cuda")

        for fpath in fpaths:

            fname = os.path.basename(fpath).split('.')[0]
            src_lang= fname[:2] if args.src_lang is None else args.src_lang
            tgt_lang = fname[2:] if args.tgt_lang is None else args.tgt_lang

            src_lang = lang_code[src_lang] if args.nllb else src_lang
            tgt_lang = lang_code[tgt_lang] if args.nllb else tgt_lang

            mt_forward_lprobs_mean, mt_backward_lprobs_mean, mt_forward_lprobs_sum, mt_backward_lprobs_sum,  mt_forward_lprobs_min, mt_backward_lprobs_min = [], [], [], [], [], []

            if args.skip_src:
                dataset = TranslationeseMTDataset(fpath, mt_tokenizer, src_lang=src_lang, tgt_lang=tgt_lang,
                                                  skip_src=True, csv_quoting=args.csv_quoting)
                column_names = ["skipped_forward_lprobs_mean", "skipped_backward_lprobs_mean", "skipped_forward_lprobs_sum",
                                "skipped_backward_lprobs_sum", "skipped_forward_lprobs_min", "skipped_backward_lprobs_min"]
            elif args.mask_src:
                dataset = TranslationeseMTDataset(fpath, mt_tokenizer, src_lang=src_lang, tgt_lang=tgt_lang,
                                                  mask_src=True, csv_quoting=args.csv_quoting)
                column_names = ["masked_forward_lprobs_mean", "masked_backward_lprobs_mean", "masked_forward_lprobs_sum",
                                "masked_backward_lprobs_sum", "masked_forward_lprobs_min", "masked_backward_lprobs_min"]
            else:
                dataset = SingleModelTranslationDataset(fpath, mt_tokenizer, src_lang=src_lang, tgt_lang=tgt_lang,
                                                        mt_model=True, csv_quoting=args.csv_quoting)
                column_names = ["mt_forward_lprobs_mean", "mt_backward_lprobs_mean", "mt_forward_lprobs_sum",
                                "mt_backward_lprobs_sum", "mt_forward_lprobs_min", "mt_backward_lprobs_min"]

            with torch.no_grad():

                dataloader = DataLoader(dataset, batch_size=args.bsz, shuffle=False, drop_last=False,
                                        collate_fn=dataset.tokenize)

                for bid, batch in enumerate(dataloader):
                    mt_forward_inputs, mt_backward_inputs = batch

                    mt_forward_labels = mt_forward_inputs.labels
                    mt_backward_labels = mt_backward_inputs.labels

                    mt_fd_lp_mean, mt_fd_lp_sum, mt_fd_lp_min = get_log_probs(mt_forward_inputs, mt_model, device, xentropy,
                                                                mt_forward_labels)
                    mt_bd_lp_mean, mt_bd_lp_sum, mt_bd_lp_min = get_log_probs(mt_backward_inputs, mt_model, device, xentropy,
                                                                mt_backward_labels)

                    mt_forward_lprobs_mean += mt_fd_lp_mean
                    mt_forward_lprobs_sum += mt_fd_lp_sum
                    mt_forward_lprobs_min += mt_fd_lp_min
                    mt_backward_lprobs_mean += mt_bd_lp_mean
                    mt_backward_lprobs_sum += mt_bd_lp_sum
                    mt_backward_lprobs_min += mt_bd_lp_min

                values = [mt_forward_lprobs_mean, mt_backward_lprobs_mean, mt_forward_lprobs_sum, mt_backward_lprobs_sum,
                          mt_forward_lprobs_min, mt_backward_lprobs_min]

                file = basename(fpath)
                fname_df[file] = fname_df.get(file, dataset.data)
                for mname, measure in zip(column_names, values):
                    fname_df[file][mname] = np.array(measure)


        for fname, df in fname_df.items():
            df.to_csv(os.path.join(args.out_dir, fname), sep='\t', index=False)

if __name__ == "__main__":
    main()




