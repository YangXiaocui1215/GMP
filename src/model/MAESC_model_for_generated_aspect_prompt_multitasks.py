from typing import Optional, Tuple
from fastNLP.modules.torch.encoder import Seq2SeqEncoder
from fastNLP.modules.torch.decoder import Seq2SeqDecoder
from fastNLP.modules.torch import State
import torch
import torch.nn.functional as F
from torch import nn
from src.model.modeling_bart import (PretrainedBartModel, BartEncoder,
                                     BartDecoder, BartModel,
                                     BartClassificationHead,
                                     _make_linear_from_emb,
                                     _prepare_bart_decoder_inputs)
from transformers import BartTokenizer

from src.model.config import MultiModalBartConfig
#from src.model.mixins import GenerationMixin, FromPretrainedMixin
from src.model.modules_for_prompt_multitasks import MultiModalBartEncoder, MultiModalBartDecoder_span, Span_loss, MultiModalBartEncoder_for_Generating_aspect_prompt, MultiModalBartDecoder_generate_aspect_prompt
from src.model.modules_for_prompt_multitasks import MultiModalBartDecoder_aspects_num 


class MultiModalBartModel_AESC(PretrainedBartModel):
    def build_model(self,
                    args,
                    bart_model,
                    tokenizer,
                    label_ids,
                    config,
                    decoder_type=None,
                    copy_gate=False,
                    use_encoder_mlp=False,
                    use_recur_pos=False,
                    tag_first=False):
        if args.bart_init:
            model = BartModel.from_pretrained(bart_model)
            num_tokens, _ = model.encoder.embed_tokens.weight.shape
            print('num_tokens', num_tokens)

            model.resize_token_embeddings(
                len(tokenizer.unique_no_split_tokens) + num_tokens)
            encoder = model.encoder
            decoder = model.decoder

            padding_idx = config.pad_token_id
            encoder.embed_tokens.padding_idx = padding_idx

            # if use_recur_pos:
            #     decoder.set_position_embedding(label_ids[0], tag_first)

            _tokenizer = BartTokenizer.from_pretrained(bart_model)

            for token in tokenizer.unique_no_split_tokens:
                if token[:2] == '<<':  # 特殊字符
                    index = tokenizer.convert_tokens_to_ids(
                        tokenizer._base_tokenizer.tokenize(token))
                    if len(index) > 1:
                        raise RuntimeError(f"{token} wrong split")
                    else:
                        index = index[0]
                    assert index >= num_tokens, (index, num_tokens, token)
                    indexes = _tokenizer.convert_tokens_to_ids(
                        _tokenizer.tokenize(token[2:-2]))
                    embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                    for i in indexes[1:]:
                        embed += model.decoder.embed_tokens.weight.data[i]
                    embed /= len(indexes)
                    model.decoder.embed_tokens.weight.data[index] = embed
        else:
            raise RuntimeError("error init!!!!!!!")

        multimodal_encoder_for_generated_aspect_prompt = MultiModalBartEncoder(config, encoder,
                                                   tokenizer.img_feat_id,
                                                   tokenizer.cls_token_id,
                                                   args.num_image_tokens)

        multimodal_encoder = MultiModalBartEncoder_for_Generating_aspect_prompt(
                                                                         use_generated_prompt=args.use_generated_prompt,
                                                                         config=config, 
                                                                         encoder = encoder,
                                                                         img_feat_id = tokenizer.img_feat_id,
                                                                         aspect_prompt_token_id=tokenizer.aspect_prompt_token_id,
                                                                         senti_prompt_token_id=tokenizer.senti_prompt_token_id,
                                                                         cls_token_id = tokenizer.cls_token_id,
                                                                         num_image_tokens = args.num_image_tokens,
                                                                         use_different_aspect_prompt=args.use_different_aspect_prompt 
                                                  
                                                   )
        return (multimodal_encoder_for_generated_aspect_prompt, multimodal_encoder, decoder)

    def __init__(self, config: MultiModalBartConfig, args, bart_model,
                 tokenizer, label_ids):
        super().__init__(config)
        self.config = config
        self.tokenizer = tokenizer
        label_ids = sorted(label_ids)
        multimodal_encoder_for_generated_aspect_prompt, multimodal_encoder, share_decoder = self.build_model(
            args, bart_model, self.tokenizer, label_ids, config)
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        self.causal_mask = causal_mask.triu(diagonal=1)
        self.use_multitasks = args.use_multitasks
        self.loss_lambda = args.loss_lambda
        self.num_image_tokens = args.num_image_tokens

        self.aspect_prompt_encoder = multimodal_encoder_for_generated_aspect_prompt
        self.encoder = multimodal_encoder

        only_sc = False
        # need_tag = True  #if predict the sentiment or not
        if args.task == 'twitter_ae':
            need_tag = False
        else:
            need_tag = True
            # if args.task == 'twitter_sc':
            #     only_sc = True

        self.prompt_decoder = MultiModalBartDecoder_generate_aspect_prompt(self.config, share_decoder)

        if self.use_multitasks:
            self.aspect_num_decoder = MultiModalBartDecoder_aspects_num(self.config, share_decoder)

        self.decoder = MultiModalBartDecoder_span(self.config,
                                                  self.tokenizer,
                                                  share_decoder,
                                                  self.tokenizer.pad_token_id,
                                                  label_ids,
                                                  self.causal_mask,
                                                  num_image_tokens=self.num_image_tokens,
                                                  need_tag=need_tag,
                                                  only_sc=False)
        self.span_loss_fct = Span_loss()

    def prepare_state(self,
                      input_ids,
                      image_features,
                      attention_mask=None,
                      aesc_infos=None,
                      aspects_num=None,
                      first=None):
        ##generate prompt for each instance

        prompt_attention_mask = attention_mask
        if self.num_image_tokens==0:
            end_index = 62
            begin_index = 22
        elif self.num_image_tokens==1:
            end_index = 63
            begin_index = 23
        elif self.num_image_tokens==2:
            end_index = 64
            begin_index = 24
        elif self.num_image_tokens==3:
            end_index = 65
            begin_index = 25
        elif self.num_image_tokens==4:
            end_index = 66
            begin_index = 26
        elif self.num_image_tokens==5:
            end_index = 67
            begin_index = 27
        elif self.num_image_tokens==6:
            end_index = 68
            begin_index = 28
        elif self.num_image_tokens==7:
            end_index = 69
            begin_index = 29
        
        
        
        for i in range(len(prompt_attention_mask)):
            mask = prompt_attention_mask[i]
            mask[begin_index:end_index]=torch.zeros_like(mask[begin_index:end_index]) ##26:66 是aspect提示的位置
            prompt_attention_mask[i]=mask
        
        ''' aspects_prompt '''
        dict_for_prompt = self.aspect_prompt_encoder(input_ids=input_ids,
                                              image_features=image_features,
                                              attention_mask=prompt_attention_mask,
                                              output_hidden_states=True,
                                              return_dict=True)

        
        aspect_prompt_decoder_input_ids, aspect_prompt_decoder_attention_mask = [
            aesc_infos['aspect_prompt_decoder_input_ids'].to(input_ids.device),
            aesc_infos['aspect_prompt_decoder_attention_mask'].to(input_ids.device)]
        generated_prompt = self.prompt_decoder(
                                            encoder_outputs=dict_for_prompt.last_hidden_state, 
                                            attention_mask=attention_mask,
                                            decoder_input_ids =aspect_prompt_decoder_input_ids, decoder_attention_mask=aspect_prompt_decoder_attention_mask)

        generated_prompt = generated_prompt[:, 1:, :] ##(batch_size, 2, 768)

        '''aspects_num'''
        aspects_num_decoder_input_ids, aspects_num_decoder_attention_mask = [
            aesc_infos['aspects_num_decoder_input_ids'].to(input_ids.device),
            aesc_infos['aspects_num_decoder_attention_mask'].to(input_ids.device)]

        # import ipdb; ipdb.set_trace()
        if self.use_multitasks:
            aspects_num_loss, predict_aspects_num_logits = self.aspect_num_decoder(aspects_num_labels=aspects_num,
                                                                            encoder_outputs=dict_for_prompt[0], 
                                                                            attention_mask=attention_mask,
                                                                            aspects_num_decoder_input_ids=aspects_num_decoder_input_ids)


            predict_aspects_num = torch.argmax(predict_aspects_num_logits, dim=1)
            new_predict_aspects_num = predict_aspects_num + torch.ones_like(predict_aspects_num)
        else:
            aspects_num_loss =0
            new_predict_aspects_num = []
            predict_aspects_num = []
            for i in range(len(input_ids)):
                new_predict_aspects_num.append(5)
                predict_aspects_num.append(4)
            new_predict_aspects_num = torch.tensor(new_predict_aspects_num)
            predict_aspects_num = torch.tensor(predict_aspects_num)

        dict = self.encoder(
                            input_ids=input_ids,
                            image_features=image_features,
                            attention_mask=attention_mask,
                            generated_prompt= generated_prompt,
                            aspects_num = new_predict_aspects_num,
                            output_hidden_states=True,
                            return_dict=True)


        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        encoder_mask = attention_mask
        src_embed_outputs = hidden_states[0]
        state = BartState(
            encoder_outputs,
            encoder_mask,
            input_ids[:,
                      end_index:],  #the text features start from index 38, the front are image features.
            first,
            src_embed_outputs)
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        return state, aspects_num_loss, predict_aspects_num


    def forward(
            self,
            input_ids,
            image_features,
            attention_mask=None,
            aesc_infos=None,
            aspects_num=None,
            encoder_outputs: Optional[Tuple] = None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        ### for prompt
        # import ipdb; ipdb.set_trace()
       
        ## for aspect-spans
      
        aspects_num = torch.tensor(aspects_num).to(input_ids.device)
        state, aspects_num_loss, predict_aspects_num = self.prepare_state( input_ids, image_features, attention_mask, aesc_infos, aspects_num)
        spans, span_mask = [ 
            aesc_infos['labels'].to(input_ids.device),
            aesc_infos['masks'].to(input_ids.device)
        ]

        logits = self.decoder(spans, state) ## spans: (2, 13) logits: (2, 12, 40)

        span_loss = self.span_loss_fct(spans[:, 1:], logits, span_mask[:, 1:])

        all_loss = span_loss + self.loss_lambda*aspects_num_loss

        return all_loss, predict_aspects_num



class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first,
                 src_embed_outputs):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs,
                                                     indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(
                                layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new
