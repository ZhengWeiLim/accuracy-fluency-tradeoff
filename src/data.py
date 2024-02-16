from torch.utils.data import Dataset
from torch.nn.functional import pad
import torch
import pandas as pd
import csv



lang_code = {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans",
             "ms": "zsm_Latn", "ja": "jpn_Jpan", "nl": "nld_Latn", "es": "spa_Latn", "ar": "arb_Arab",
             "da": "dan_Latn", "hi": "hin_Deva", "pt": "por_Latn", "et": "est_Latn", "po": "pol_Latn",
             "pl": "pol_Latn", "ro": "ron_Latn", "ne": "npi_Deva", "si": "sin_Sinh", "mr": "mar_Deva",
             "cs": "ces_Latn", "km": "khm_Khmr", "ps": "pbt_Arab", "fi": "fin_Latn", "tr": "tur_Latn"}

lang_name = {"en": "English", "de": "German", "zh": "Chinese", "ru": "Russian", "tr": "Turkish", "cs": "Czech", "fi": 'Finnish',
             "ro": "Romanian"}


def mask_input(inputs, tokenizer):
    masked = inputs['input_ids'] * inputs['special_tokens_mask'] # e.g., [special token, 0, 0, ... ]
    masked[masked == 0] = tokenizer.mask_token_id #  [special token, maskid, maskid, ... ]
    inputs['input_ids'] = masked
    return inputs

class SingleModelTranslationDataset(Dataset):
    def __init__(self, fpath, tokenizer, input_split_into_words=False, mt_model=False,
                 src_lang=None, tgt_lang=None, csv_quoting=csv.QUOTE_NONE):
        self.data = pd.read_csv(fpath, sep='\t', on_bad_lines='warn', quoting=csv_quoting)
        self.tokenizer = tokenizer
        self.input_split_into_words = input_split_into_words
        self.mt_model = mt_model
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row.source.strip().replace('<v>', '').replace('</v>', ''), row.target.strip().replace('<v>', '').replace('</v>', '')

    def tokenize(self, lines):
        source, target = zip(*lines)
        if not self.mt_model:
            src_inputs = self.tokenizer(source, is_split_into_words=self.input_split_into_words,
                                        return_tensors='pt', padding=True)
            tgt_inputs = self.tokenizer(target, is_split_into_words=self.input_split_into_words,
                                        return_tensors='pt', padding=True)
            return src_inputs, tgt_inputs
        else:
            self.tokenizer.src_lang, self.tokenizer.tgt_lang = self.src_lang, self.tgt_lang
            mt_forward_inputs = self.tokenizer(source, text_target=target,
                                                  is_split_into_words=self.input_split_into_words,
                                                  return_tensors="pt", padding=True)

            self.tokenizer.src_lang, self.tokenizer.tgt_lang = self.tgt_lang, self.src_lang
            mt_backward_inputs = self.tokenizer(target, text_target=source,
                                                   is_split_into_words=self.input_split_into_words,
                                                   return_tensors="pt", padding=True)
            return mt_forward_inputs, mt_backward_inputs


class SingleModelCRITTDataset(SingleModelTranslationDataset):
    def __init__(self, fpath, tokenizer, mt_model=False, input_split_into_words=True, mask_src=False, skip_src=False,
                 use_lang_code=True):
        super().__init__(fpath, tokenizer, input_split_into_words=input_split_into_words, mt_model=mt_model,
                 src_lang=None, tgt_lang=None)
        self.data['source'] = self.data['src_tokens'].apply(lambda x: ' '.join(x.split('_')))
        self.data['target'] = self.data['tgt_tokens'].apply(lambda x: ' '.join(x.split('_')))
        self.skip_src = skip_src
        self.mask_src = mask_src
        self.use_lang_code = use_lang_code


    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if self.use_lang_code:
            src_lang, tgt_lang =  lang_code[row.src_lang], lang_code[row.tgt_lang]
        else:
            src_lang, tgt_lang = row.src_lang, row.tgt_lang
        return src_lang, tgt_lang, row.source.strip(), row.target.strip()


    def tokenize(self, line):
        src_lang, tgt_lang, source, target = zip(*line)
        if self.mt_model:

            src_input = ["" for _ in source] if self.skip_src else source
            tgt_input = ["" for _ in target] if self.skip_src else target

            self.tokenizer.src_lang, self.tokenizer.tgt_lang = src_lang[0], tgt_lang[0]
            mt_forward_input = self.tokenizer(src_input, text_target=target, is_split_into_words=self.input_split_into_words,
                                                  return_tensors="pt", padding=True, return_special_tokens_mask=True)

            self.tokenizer.src_lang, self.tokenizer.tgt_lang = tgt_lang[0], src_lang[0]
            mt_backward_input = self.tokenizer(tgt_input, text_target=source, is_split_into_words=self.input_split_into_words,
                                                   return_tensors="pt", padding=True, return_special_tokens_mask=True)

            if self.mask_src:
                mt_forward_input = mask_input(mt_forward_input, self.tokenizer)
                mt_backward_input = mask_input(mt_backward_input, self.tokenizer)

            del mt_forward_input['special_tokens_mask']
            del mt_backward_input['special_tokens_mask']

            return mt_forward_input, mt_backward_input
        else:
            src_input = self.tokenizer(source, return_tensors="pt", padding=True, is_split_into_words=self.input_split_into_words)
            tgt_input = self.tokenizer(target, return_tensors="pt", padding=True, is_split_into_words=self.input_split_into_words)

            return src_input, tgt_input


class TranslationeseMTDataset(SingleModelTranslationDataset):
    def __init__(self, fpath, tokenizer, input_split_into_words=False, src_lang=None, tgt_lang=None, mask_src=False, skip_src=False,
                 csv_quoting=csv.QUOTE_NONE):
        super().__init__(fpath, tokenizer, input_split_into_words=input_split_into_words, mt_model=True,
                 src_lang=src_lang, tgt_lang=tgt_lang, csv_quoting=csv_quoting)
        self.mask_src = mask_src
        self.skip_src = skip_src


    def tokenize(self, lines):
        source, target = zip(*lines)
        self.tokenizer.src_lang, self.tokenizer.tgt_lang = self.src_lang, self.tgt_lang

        src_input = ["" for _ in source] if self.skip_src else source
        tgt_input = ["" for _ in target] if self.skip_src else target

        mt_forward_inputs = self.tokenizer(src_input, text_target=target,
                                           is_split_into_words=self.input_split_into_words,
                                           return_tensors="pt", padding=True, return_special_tokens_mask=True)

        self.tokenizer.src_lang, self.tokenizer.tgt_lang = self.tgt_lang, self.src_lang
        mt_backward_inputs = self.tokenizer(tgt_input, text_target=source,
                                            is_split_into_words=self.input_split_into_words,
                                            return_tensors="pt", padding=True, return_special_tokens_mask=True)

        if self.mask_src:
            mt_forward_inputs = mask_input(mt_forward_inputs, self.tokenizer)
            mt_backward_inputs = mask_input(mt_backward_inputs, self.tokenizer)

        del mt_forward_inputs['special_tokens_mask']
        del mt_backward_inputs['special_tokens_mask']

        return mt_forward_inputs, mt_backward_inputs






class TranslationDataset(Dataset):
    def __init__(self, fpath, mt_tokenizer, lm_tokenizer, src_lang, tgt_lang, truncate=False,
                 input_split_into_words=False):
        self.data = pd.read_csv(fpath, sep='\t', on_bad_lines='warn', quoting=csv.QUOTE_NONE)
        self.mt_tokenizer = mt_tokenizer
        self.lm_tokenizer = lm_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.truncate = truncate
        self.max_length=self.mt_tokenizer.model_max_length
        self.input_split_into_words = input_split_into_words

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row.source.strip(), row.target.strip()

    def tokenize(self, lines):
        source, target = zip(*lines)
        self.mt_tokenizer.src_lang, self.mt_tokenizer.tgt_lang = self.src_lang, self.tgt_lang
        mt_forward_inputs = self.mt_tokenizer(source, text_target=target, is_split_into_words=self.input_split_into_words,
                                              return_tensors="pt", padding=True, truncation=self.truncate)

        self.mt_tokenizer.src_lang, self.mt_tokenizer.tgt_lang = self.tgt_lang, self.src_lang
        mt_backward_inputs = self.mt_tokenizer(target, text_target=source, is_split_into_words=self.input_split_into_words,
                                               return_tensors="pt", padding=True, truncation=self.truncate)

        lm_src_inputs = self.lm_tokenizer(source, return_tensors="pt", padding=True, return_attention_mask=True,
                                          is_split_into_words=self.input_split_into_words, truncation=self.truncate,
                                          max_length=self.max_length)
        lm_tgt_inputs = self.lm_tokenizer(target, return_tensors="pt", padding=True, return_attention_mask=True,
                                          is_split_into_words=self.input_split_into_words, truncation=self.truncate,
                                          max_length=self.max_length)

        return mt_forward_inputs, mt_backward_inputs, lm_src_inputs, lm_tgt_inputs





class CRITTTranslationDataset(TranslationDataset):
    def __init__(self, fpath, mt_tokenizer, lm_tokenizer, src_lang, tgt_lang, truncate=False, input_split_into_words=True):
        super().__init__(fpath, mt_tokenizer, lm_tokenizer, src_lang, tgt_lang, truncate, input_split_into_words)
        self.data['source'] = self.data['src_tokens'].apply(lambda x: ' '.join(x.split('_')))
        self.data['target'] = self.data['tgt_tokens'].apply(lambda x: ' '.join(x.split('_')))


    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return lang_code[row.src_lang], lang_code[row.tgt_lang], row.source.strip(), row.target.strip()


    def tokenize(self, line):
        src_lang, tgt_lang, source, target = zip(*line)
        self.mt_tokenizer.src_lang, self.mt_tokenizer.tgt_lang = src_lang[0], tgt_lang[0]
        mt_forward_input = self.mt_tokenizer(source, text_target=target, is_split_into_words=self.input_split_into_words,
                                              return_tensors="pt", padding=True, truncation=self.truncate)

        self.mt_tokenizer.src_lang, self.mt_tokenizer.tgt_lang = tgt_lang[0], src_lang[0]
        mt_backward_input = self.mt_tokenizer(target, text_target=source, is_split_into_words=self.input_split_into_words,
                                               return_tensors="pt", padding=True, truncation=self.truncate)

        lm_src_input = self.lm_tokenizer(source, return_tensors="pt", padding=True, return_attention_mask=True,
                                          is_split_into_words=self.input_split_into_words, truncation=self.truncate,
                                          max_length=self.max_length)
        lm_tgt_input = self.lm_tokenizer(target, return_tensors="pt", padding=True, return_attention_mask=True,
                                          is_split_into_words=self.input_split_into_words, truncation=self.truncate,
                                          max_length=self.max_length)

        return mt_forward_input, mt_backward_input, lm_src_input, lm_tgt_input



class TranslationDatasetLLM(TranslationDataset):
    def __init__(self, fpath, tokenizer, src_lang, tgt_lang, truncate=False, input_split_into_words=False, left_padding=True):
        super().__init__(fpath, tokenizer, tokenizer, src_lang, tgt_lang, truncate, input_split_into_words)
        self.tokenizer = tokenizer
        self.left_padding = left_padding

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:  source, target, forward_prompt, backward_prompt, forward_prompt_with_trans, backward_prompt_with_trans
        '''
        pass

    def tokenize(self, lines):
        source, target, forward_prompt, backward_prompt, forward_mt, backward_mt = zip(*lines)

        src_input, tgt_input, forward_prompt_input, backward_prompt_input, forward_mt_input, backward_mt_input = (
            self.tokenizer(text, is_split_into_words=self.input_split_into_words, return_tensors="pt", padding=True,
                           truncation=self.truncate)
            for text in [source, target, forward_prompt, backward_prompt, forward_mt, backward_mt])

        if self.left_padding:
            forward_prompt_mask = tgt_input['input_ids'] != self.tokenizer.pad_token_id
            forward_padding_size = forward_mt_input['input_ids'].size(-1) - tgt_input['input_ids'].size(-1)
            forward_prompt_mask = pad(forward_prompt_mask.int(), (forward_padding_size, 0), 'constant', 0)
            backward_prompt_mask = src_input['input_ids'] != self.tokenizer.pad_token_id
            backward_padding_size = backward_mt_input['input_ids'].size(-1) - src_input['input_ids'].size(-1)
            backward_prompt_mask = pad(backward_prompt_mask.int(), (backward_padding_size, 0), 'constant', 0)
        else:
            forward_prompt_mask = forward_prompt_input['input_ids'] == self.tokenizer.pad_token_id
            forward_padding_size = forward_mt_input['input_ids'].size(-1) - forward_prompt_input['input_ids'].size(-1)
            forward_prompt_mask = pad(forward_prompt_mask.int(), (0, forward_padding_size), 'constant', 1)
            backward_prompt_mask = backward_prompt_input['input_ids'] == self.tokenizer.pad_token_id
            backward_padding_size = backward_mt_input['input_ids'].size(-1) - backward_prompt_input['input_ids'].size(-1)
            backward_prompt_mask = pad(backward_prompt_mask.int(), (0, backward_padding_size), 'constant', 1)

        return forward_mt_input, backward_mt_input, src_input, tgt_input, forward_prompt_mask, backward_prompt_mask


class TranslationDatasetBloomz(TranslationDatasetLLM):
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src, tgt = row.source.strip(), row.target.strip()
        ended_source = src if src[-1] == '.' else src + '.'
        ended_target = tgt if tgt[-1] == '.' else tgt + '.'
        forward_prompt = f"Translate to {lang_name[self.tgt_lang]}: {ended_source} Translation:"
        backward_prompt = f"Translate to {lang_name[self.src_lang]}: {ended_target} Translation:"
        forward_mt = f"{forward_prompt} {row.target}"
        backward_mt = f"{backward_prompt} {row.source}"
        return src, tgt, forward_prompt, backward_prompt, forward_mt, backward_mt


class TranslationDatasetALMA(TranslationDatasetLLM):
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src, tgt = row.source.strip(), row.target.strip()
        src_lang, tgt_lang = lang_name[self.src_lang], lang_name[self.tgt_lang]
        forward_prompt = f"Translate this from {src_lang} to {tgt_lang}:\n{src_lang}: {row.source}\n{tgt_lang}:"
        backward_prompt = f"Translate this from {tgt_lang} to {src_lang}:\n{tgt_lang}: {row.target}\n{src_lang}:"
        forward_mt = f"{forward_prompt} {tgt}"
        backward_mt = f"{backward_prompt} {src}"
        return src, tgt, forward_prompt, backward_prompt, forward_mt, backward_mt

    def tokenize(self, lines):
        source, target, forward_prompt, backward_prompt, forward_mt, backward_mt = zip(*lines)

        src_input, tgt_input, forward_prompt_input, backward_prompt_input, forward_mt_input, backward_mt_input = (
            self.tokenizer(text, is_split_into_words=self.input_split_into_words, return_tensors="pt", padding=True,
                           truncation=self.truncate)
            for text in [source, target, forward_prompt, backward_prompt, forward_mt, backward_mt])


        forward_prompt_mask = (tgt_input['input_ids'] != self.tokenizer.pad_token_id) &  (tgt_input['input_ids'] != self.tokenizer.bos_token_id)
        forward_padding_size = forward_mt_input['input_ids'].size(-1) - tgt_input['input_ids'].size(-1)
        forward_prompt_mask = pad(forward_prompt_mask.int(), (forward_padding_size, 0), 'constant', 0)
        backward_prompt_mask = src_input['input_ids'] != self.tokenizer.pad_token_id  &  (src_input['input_ids'] != self.tokenizer.bos_token_id)
        backward_padding_size = backward_mt_input['input_ids'].size(-1) - src_input['input_ids'].size(-1)
        backward_prompt_mask = pad(backward_prompt_mask.int(), (backward_padding_size, 0), 'constant', 0)

        return forward_mt_input, backward_mt_input, src_input, tgt_input, forward_prompt_mask, backward_prompt_mask



class ReferenceBasedTranslationDataset(Dataset):

    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row.target, row.ref




