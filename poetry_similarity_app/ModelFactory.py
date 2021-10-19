import numpy as np
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch


class ModelFactory:
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls._MODEL_TYPE] = cls

    @classmethod
    def create(cls, model_dir):
        model_type = ModelFactory.detect_model_type(model_dir)
        if model_type not in cls.subclasses:
            raise ValueError('Invalid model type {}'.format(model_type))

        return cls.subclasses[model_type](model_dir)

    @staticmethod
    def detect_model_type(model_dir: str) -> str:
        model_name = model_dir.strip('/')
        model_type = model_name.split('-')[0]
        return model_type


class RobertaModel(ModelFactory):
    _MODEL_TYPE = 'roberta'

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model_path = self.get_model_path()
        self.model = self.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, config=self.model.config)
        self.allowed_composition_functions = ["sum", "ind_rl"]
        self.vector_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

    def get_model_path(self):
        if self.model_dir == 'roberta-m':
            return 'sentence-transformers/stsb-xlm-r-multilingual'
        elif self.model_dir == 'roberta-alberti':
            return 'flax-community/alberti-bert-base-multilingual-cased'
        else:
            return os.path.abspath(f'models/{self.model_dir}/')

    def load_model(self):
        config = AutoConfig.from_pretrained(self.model_path, output_hidden_states=True, output_attentions=True)
        model = AutoModel.from_pretrained(self.model_path, config=config)
        return model

    def make_attention_mask_without_special_token(self, attention_mask):
        attention_mask_without_special_tok = attention_mask.clone().detach()
        attention_mask_without_special_tok[:, 0] = 0
        sent_len = attention_mask_without_special_tok.sum(1).tolist()
        col_idx = torch.LongTensor(sent_len)
        row_idx = torch.arange(attention_mask.size(0)).long()
        attention_mask_without_special_tok[row_idx, col_idx] = 0
        return attention_mask_without_special_tok

    def mean_pooling_no_spec_tokens(self, model_output, attention_mask):
        attention_mask = self.make_attention_mask_without_special_token(attention_mask)
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def compute_token_embeddings(self, text: str) -> np.array:
        if not text.strip():
            raise ValueError('No input text')
        encoded_input = self.tokenizer(text, padding=False, truncation=True, add_special_tokens=True, max_length=500,
                                       return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            token_embeddings = model_output[0][0][1:-1]
        token_embeddings = token_embeddings.cpu().detach().numpy()
        return list(token_embeddings)

    def compute_sentence_embeddings(self, text) -> np.array:
        if not text.strip():
            raise ValueError('No input text')
        encoded_input = self.tokenizer(text, padding=True, truncation=True, add_special_tokens=True, max_length=128,
                                       return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling_no_spec_tokens(model_output, encoded_input['attention_mask'])
        return sentence_embeddings.cpu().detach().numpy()[0]
