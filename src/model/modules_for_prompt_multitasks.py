import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import *
from src.model.modeling_bart import (
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    invert_mask,
    EncoderLayer,
    LayerNorm,
)
from src.model.modeling_bart import (PretrainedBartModel, BartDecoder,
                                     BartClassificationHead,
                                     _make_linear_from_emb,
                                     _prepare_bart_decoder_inputs)
from src.model.config import MultiModalBartConfig

from transformers import AutoConfig, AutoModel, CLIPVisionModel, CLIPVisionConfig
import timm
from src.model.attention import Attention_for_Senti_Prompt

TIMM_MODELS = {
    'nf_resnet50': 2048,
}
def is_clip_model(model_name):
    return model_name.startswith('openai/clip-')

image_model_name =  'nf_resnet50'
if image_model_name in TIMM_MODELS.keys():
    image_encoder = timm.create_model(image_model_name, pretrained=True, num_classes=0)
elif is_clip_model(image_model_name):
    ###model_name ='openai/clip-vit-base-patch32'
    config = CLIPVisionConfig.from_pretrained(image_model_name)
    image_encoder = CLIPVisionModel.from_pretrained(
            image_model_name,
            config=config,
        )
else:
    image_encoder = AutoModel.from_pretrained(image_model_name)

def init_image_encoder(image_model_name, frozen_image_encoder, num_image_tokens, d_text_encoder):

    # image_encoder = get_image_encoder(image_model_name)
    d_image_encoder = _d_image_encoder(image_model_name, image_encoder)

    if frozen_image_encoder:
        for p in image_encoder.parameters():
            p.requires_grad = False
            image_encoder.eval()

    proj_image_features = nn.Linear(
            in_features=d_image_encoder,
            out_features=num_image_tokens * d_text_encoder,
        )
    return proj_image_features.cuda(), d_image_encoder

def _d_image_encoder(image_model_name, image_encoder):
    ##image_model_name默认为： 'microsoft/resnet-50'
    model_name = image_model_name
    if model_name in TIMM_MODELS.keys():
        return TIMM_MODELS[model_name]
    elif is_clip_model(model_name):
        return image_encoder.config.hidden_size
    elif model_name.startswith('microsoft/resnet-'):
        return image_encoder.config.hidden_sizes[-1]
    else:
        return image_encoder.config.hidden_size


def encode_images(image_encoder, proj_image_features, frozen_image_encoder, pixel_values, d_image_encoder):
    
    image_encoder = image_encoder.cuda()
    pixel_values = pixel_values.cuda()
    # print('the shape of pixel_values is {}'.format(pixel_values.shape))
    batch_size = pixel_values.shape[0]

    if frozen_image_encoder:
        with torch.no_grad():
            image_encoder.eval()
            visual = image_encoder(pixel_values)
    else:
        visual = image_encoder(pixel_values)

    if not isinstance(visual, torch.Tensor):  # HuggingFace model
        visual = visual.pooler_output

    visual = visual.reshape(batch_size, d_image_encoder)
    visual = proj_image_features(visual).cuda()
    return visual



class ImageEmbedding(nn.Module):
    def __init__(self, image_dim, final_dim, image_model_name, frozen_image_encoder=False, num_image_tokens=2):
        super(ImageEmbedding, self).__init__()
        self.frozen_image_encoder = frozen_image_encoder
        self.final_dim = final_dim
        self.linear = nn.Linear(final_dim, final_dim)
        self.d_image_encoder = _d_image_encoder(image_model_name, image_encoder)
        if frozen_image_encoder:
            for p in image_encoder.parameters():
                p.requires_grad = False
                image_encoder.eval()

        self.proj_image_features = nn.Linear(
                in_features=self.d_image_encoder,
                out_features=num_image_tokens * final_dim,
            )

    def forward(self, image_pixel_values):
        
        # import ipdb; ipdb.set_trace()
        image_pixel_values = torch.stack(image_pixel_values)
        batch_size = image_pixel_values.size(0)
        image_features = encode_images(image_encoder=image_encoder, 
                                           proj_image_features=self.proj_image_features, 
                                           frozen_image_encoder=self.frozen_image_encoder, 
                                           pixel_values=image_pixel_values, 
                                           d_image_encoder=self.d_image_encoder)
        ###image_features: (batch_size, num_image_tokens*1024) (4, 2048)
        # print("======================================the shape of image_features is {}=====================================".format(image_features.shape))
        image_features = image_features.reshape(batch_size, -1, self.final_dim) ### (4, num_image_tokens, 1024(d_model))
        
        img_len = list(map(len, image_features))
        non_empty_features = list(filter(lambda x: len(x) > 0, image_features))

        embedded = None
        if len(non_empty_features) > 0:
            img_tensor = torch.cat(non_empty_features, dim=0)
            embedded = self.linear(img_tensor)

        output = []
        index = 0
        for l in img_len:
            if l > 0:
                output.append(embedded[index:index + l])
            else:
                output.append(torch.empty(0))
            index += l
        return output


class MultiModalBartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:EncoderLayer.

    Args:
        config: MultiModalBartConfig
    """
    def __init__(self, config: MultiModalBartConfig, encoder, img_feat_id,
                 cls_token_id, num_image_tokens):
        super().__init__()

        self.img_feat_id = img_feat_id
        self.cls_token_id = cls_token_id
        embed_tokens = encoder.embed_tokens
        self.dropout = encoder.dropout
        self.layerdrop = encoder.layerdrop

        self.indentity = nn.Identity()

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = encoder.embed_scale
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = encoder.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_images = ImageEmbedding(embed_dim, embed_dim, image_model_name, num_image_tokens=num_image_tokens)
        self.embed_positions = encoder.embed_positions

        self.layers = encoder.layers
        self.layernorm_embedding = encoder.layernorm_embedding
        # mbart has one extra layer_norm
        self.layer_norm = encoder.layer_norm

    def _embed_multi_modal(self, input_ids, image_features):
        """embed textual and visual inputs and combine them into one embedding"""
        mask = (input_ids == self.img_feat_id) | (
            input_ids == self.cls_token_id)
        # print(mask.shape)
        embedded_images = self.embed_images(image_features)
        embedded = self.embed_tokens(input_ids)
        # print('mask shape', mask.shape)
        if not embedded_images[0].dtype == torch.float32:
            embedded = embedded.half()

        for index, value in enumerate(embedded_images):
            if len(value) > 0:
                embedded[index, mask[index]] = value

        return embedded

    def forward(self,
                input_ids,
                image_features,
                attention_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param output_attentions:
        :param output_hidden_states:
        :return: Tuple comprised of:
            - x (Tensor): the last encoder layer's output of
              shape (src_len, batch, embed_dim)
            - encoder_states (List[Tensor]): all intermediate
              hidden states of shape (src_len, batch, embed_dim).
              Only populated if output_hidden_states: is True.
            - all_attentions (List[Tensor]): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self._embed_multi_modal(
            input_ids, image_features) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability <
                                  self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x,
                                        attention_mask,
                                        output_attentions=output_attentions)

            if output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [
            hidden_state.transpose(0, 1) for hidden_state in encoder_states
        ]
        x = x.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions]
                         if v is not None)
        return BaseModelOutput(last_hidden_state=x,
                               hidden_states=encoder_states,
                               attentions=all_attentions)

class MultiModalBartEncoder_for_Generating_aspect_prompt(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:EncoderLayer.

    Args:
        config: MultiModalBartConfig
    """
    def __init__(self, 
                use_generated_prompt,
                config: MultiModalBartConfig, encoder, img_feat_id, aspect_prompt_token_id, senti_prompt_token_id,
                 cls_token_id, num_image_tokens, use_different_aspect_prompt):
        super().__init__()

        self.use_generated_prompt= use_generated_prompt
        self.aspect_prompt_token_id = aspect_prompt_token_id
        self.senti_prompt_token_id = senti_prompt_token_id
        self.use_different_aspect_prompt = use_different_aspect_prompt

        self.img_feat_id = img_feat_id
        self.cls_token_id = cls_token_id
        embed_tokens = encoder.embed_tokens
        self.dropout = encoder.dropout
        self.layerdrop = encoder.layerdrop
        self.num_image_tokens = num_image_tokens

        self.indentity = nn.Identity()

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = encoder.embed_scale
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = encoder.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_images = ImageEmbedding(embed_dim, embed_dim, image_model_name, num_image_tokens=num_image_tokens)
        self.embed_positions = encoder.embed_positions

        self.layers = encoder.layers
        self.layernorm_embedding = encoder.layernorm_embedding
        # mbart has one extra layer_norm
        self.layer_norm = encoder.layer_norm

        # self.aspect_linear = nn.Linear(768, 768)
        # self.aspect_relu = nn.LeakyReLU()

    def _embed_multi_modal(self, generated_aspect_prompt, aspects_num, input_ids, image_features):
        """embed textual and visual inputs and combine them into one embedding"""
        # import ipdb; ipdb.set_trace()
        mask = (input_ids == self.img_feat_id) | (
            input_ids == self.cls_token_id)
        # print(mask.shape)
        embedded_images = self.embed_images(image_features)
        embedded = self.embed_tokens(input_ids)
        # print('mask shape', mask.shape)
        if not embedded_images[0].dtype == torch.float32:
            embedded = embedded.half()

        for index, value in enumerate(embedded_images):
            if len(value) > 0:
                embedded[index, mask[index]] = value
        
        new_input_ids =[]
        # import ipdb; ipdb.set_trace()
        for i in range(len(aspects_num)):

            
            aspect_num = aspects_num[i]
            # print('the aspect_num is {}'.format(aspect_num))
           
            input_id = input_ids[i]

            if self.num_image_tokens==0:
                prompt_begin_index = 25
                prompt_end_index = 39
            elif self.num_image_tokens==1:
                prompt_begin_index = 26
                prompt_end_index = 40
            elif self.num_image_tokens==2:
                prompt_begin_index = 27
                prompt_end_index = 41
            elif self.num_image_tokens==3:
                prompt_begin_index = 28
                prompt_end_index = 42
            elif self.num_image_tokens==4:
                prompt_begin_index = 29
                prompt_end_index = 43
            elif self.num_image_tokens==5:
                prompt_begin_index = 30
                prompt_end_index = 44
            elif self.num_image_tokens==6:
                prompt_begin_index = 31
                prompt_end_index = 45
            elif self.num_image_tokens==7:
                prompt_begin_index = 32
                prompt_end_index = 46
                
            # print('before')
            # print(len(input_id))
            # import ipdb; ipdb.set_trace()
            reserve_aspect_id = input_id[prompt_begin_index:prompt_begin_index+3*aspect_num]
            if aspect_num ==5:
                # print('aspect_num is 5')
                # print(reserve_aspect_id)
                new_input_id = torch.cat([input_id[:prompt_begin_index], reserve_aspect_id, input_id[prompt_end_index+1:]])
            else:
                cut_aspect_id = torch.ones_like(input_id[prompt_begin_index+3*aspect_num:prompt_end_index])
                new_input_id = torch.cat([input_id[:prompt_begin_index], reserve_aspect_id, cut_aspect_id, input_id[prompt_end_index:]])
            # print("++++++++++++++++++++cut_aspect_id++++++++++++++++++++++++")
            # print(cut_aspect_id)
            # print(input_id[58:])
            new_input_ids.append(new_input_id)
            # print('the shape of new_input_id is {}'.format(new_input_id.shape))
            # print(new_input_id[58:])
            # print("+++++++++++++++++++++++input_id length is {}+++++++++++++++++++++++".format(len(input_id)))
            # print(input_id)
            # print("+++++++++++++++++++++++new_input_id length is {}+++++++++++++++++++++++".format(len(input_id)))
            # print(new_input_id)
        new_input_ids = torch.stack(new_input_ids)

        prompt_mask = (new_input_ids == self.aspect_prompt_token_id)

        if self.use_generated_prompt:
            if self.use_different_aspect_prompt:
                # self.aspect_linear = self.aspect_linear.to(generated_aspect_prompt.device)
                # self.aspect_relu = self.aspect_relu.to(generated_aspect_prompt.device)
            
                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]
                    # prompt_embedding_ = generated_prompt[index].repeat(aspect_num, 1)
                    prompt_embedding_list = []
                    for j in range(aspect_num):
                        aspect_linear = nn.Linear(768, 768).to(generated_aspect_prompt.device) ##每个aspect有自己的变换，为每个aspect设计特定的prompt
                        aspect_relu = nn.LeakyReLU().to(generated_aspect_prompt.device)
                        prompt_embedding = aspect_linear(generated_aspect_prompt[index])
                        prompt_embedding = aspect_relu(prompt_embedding)
                        ###可以加入激活函数
                        # prompt_embedding = nn.LeakyReLU(prompt_embedding)
                        prompt_embedding_list.append(prompt_embedding)
                    prompt_embedding_ = torch.cat(prompt_embedding_list, dim=0)
                    embedded[index, prompt_mask[index]] = prompt_embedding_
            else:
                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]
                    prompt_embedding_ = generated_aspect_prompt[index].repeat(aspect_num, 1)

                    embedded[index, prompt_mask[index]] = prompt_embedding_
        return embedded
       

    def forward(self,
                input_ids,
                image_features,
                attention_mask=None,
                generated_prompt=None,
                aspects_num=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param output_attentions:
        :param output_hidden_states:
        :return: Tuple comprised of:
            - x (Tensor): the last encoder layer's output of
              shape (src_len, batch, embed_dim)
            - encoder_states (List[Tensor]): all intermediate
              hidden states of shape (src_len, batch, embed_dim).
              Only populated if output_hidden_states: is True.
            - all_attentions (List[Tensor]): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self._embed_multi_modal(generated_prompt, aspects_num,
            input_ids, image_features) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability <
                                  self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x,
                                        attention_mask,
                                        output_attentions=output_attentions)

            if output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [
            hidden_state.transpose(0, 1) for hidden_state in encoder_states
        ]
        x = x.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions]
                         if v is not None)
        return BaseModelOutput(last_hidden_state=x,
                               hidden_states=encoder_states,
                               attentions=all_attentions)


class MultiModalBartEncoder_for_Generating_sentiment_prompt(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:EncoderLayer.

    Args:
        config: MultiModalBartConfig
    """
    def __init__(self, use_generated_prompt,
                 config: MultiModalBartConfig, encoder, img_feat_id, aspect_prompt_token_id, senti_prompt_token_id,
                 cls_token_id, num_image_tokens, use_different_senti_prompt):
        super().__init__()
        
        self.use_generated_prompt = use_generated_prompt
        self.aspect_prompt_token_id = aspect_prompt_token_id
        self.senti_prompt_token_id = senti_prompt_token_id
        self.use_different_senti_prompt = use_different_senti_prompt

        self.img_feat_id = img_feat_id
        self.cls_token_id = cls_token_id
        embed_tokens = encoder.embed_tokens
        self.dropout = encoder.dropout
        self.layerdrop = encoder.layerdrop

        self.indentity = nn.Identity()

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = encoder.embed_scale
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = encoder.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_images = ImageEmbedding(embed_dim, embed_dim, image_model_name, num_image_tokens=num_image_tokens)
        self.embed_positions = encoder.embed_positions

        self.layers = encoder.layers
        self.layernorm_embedding = encoder.layernorm_embedding
        # mbart has one extra layer_norm
        self.layer_norm = encoder.layer_norm

        # self.aspect_linear = nn.Linear(768, 768)
        # self.aspect_relu = nn.LeakyReLU()

    def _embed_multi_modal(self, generated_senti_prompt, aspects_num, input_ids, image_features):
        """embed textual and visual inputs and combine them into one embedding"""
        # import ipdb; ipdb.set_trace()
        mask = (input_ids == self.img_feat_id) | (
            input_ids == self.cls_token_id)
        # print(mask.shape)
        embedded_images = self.embed_images(image_features)
        embedded = self.embed_tokens(input_ids)
        # print('mask shape', mask.shape)
        if not embedded_images[0].dtype == torch.float32:
            embedded = embedded.half()

        for index, value in enumerate(embedded_images):
            if len(value) > 0:
                embedded[index, mask[index]] = value
        
       
        
        if self.use_generated_prompt:
            senti_prompt_mask = (input_ids == self.senti_prompt_token_id)
            # import ipdb; ipdb.set_trace()
            if self.use_different_senti_prompt:
                # self.aspect_linear = self.aspect_linear.to(generated_senti_prompt.device)
                # self.aspect_relu = self.aspect_relu.to(generated_senti_prompt.device)
            
                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]
                    # prompt_embedding_ = generated_prompt[index].repeat(aspect_num, 1)
                    prompt_embedding_list = []
                    for j in range(aspect_num):
                        aspect_linear = nn.Linear(768, 768).to(generated_senti_prompt.device)
                        aspect_relu = nn.LeakyReLU().to(generated_senti_prompt.device)
                        prompt_embedding = aspect_linear(generated_senti_prompt[index])
                        prompt_embedding = aspect_relu(prompt_embedding)
                        ###可以加入激活函数å
                        # prompt_embedding = nn.LeakyReLU(prompt_embedding)
                        prompt_embedding_list.append(prompt_embedding)
                    prompt_embedding_ = torch.cat(prompt_embedding_list, dim=0)
                    embedded[index, senti_prompt_mask[index]] = prompt_embedding_
            else:
                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]
                    prompt_embedding_ = generated_senti_prompt[index].repeat(aspect_num, 1)

                    embedded[index, senti_prompt_mask[index]] = prompt_embedding_
        return embedded
       

    def forward(self,
                input_ids,
                image_features,
                attention_mask=None,
                generated_prompt=None,
                aspects_num=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param output_attentions:
        :param output_hidden_states:
        :return: Tuple comprised of:
            - x (Tensor): the last encoder layer's output of
              shape (src_len, batch, embed_dim)
            - encoder_states (List[Tensor]): all intermediate
              hidden states of shape (src_len, batch, embed_dim).
              Only populated if output_hidden_states: is True.
            - all_attentions (List[Tensor]): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self._embed_multi_modal(generated_prompt, aspects_num,
            input_ids, image_features) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability <
                                  self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x,
                                        attention_mask,
                                        output_attentions=output_attentions)

            if output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [
            hidden_state.transpose(0, 1) for hidden_state in encoder_states
        ]
        x = x.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions]
                         if v is not None)
        return BaseModelOutput(last_hidden_state=x,
                               hidden_states=encoder_states,
                               attentions=all_attentions)


class MultiModalBartEncoder_for_Generating_Dual_prompts(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:EncoderLayer.

    Args:
        config: MultiModalBartConfig
    """
    def __init__(self, 
                 use_generated_aspect_prompt, use_generated_senti_prompt, 
                 config: MultiModalBartConfig, encoder, img_feat_id, aspect_prompt_token_id, senti_prompt_token_id,
                 cls_token_id, num_image_tokens, use_different_aspect_prompt, use_different_senti_prompt, 
                 NEU_id, POS_id, NEG_id):
        super().__init__()

        self.use_generated_aspect_prompt= use_generated_aspect_prompt
        self.use_generated_senti_prompt = use_generated_senti_prompt

        self.aspect_prompt_token_id = aspect_prompt_token_id
        self.senti_prompt_token_id = senti_prompt_token_id
        self.use_different_aspect_prompt = use_different_aspect_prompt
        self.use_different_senti_prompt = use_different_senti_prompt

        # if self.use_different_senti_prompt:
        #     self.attention_for_senti_prompt = Attention_for_Senti_Prompt(n_head=8, model_dim=768, drop_rate=0.2)
        self.img_feat_id = img_feat_id
        self.cls_token_id = cls_token_id
        self.neu_id = NEU_id
        self.pos_id = POS_id
        self.neg_id = NEG_id
        embed_tokens = encoder.embed_tokens
        self.dropout = encoder.dropout
        self.layerdrop = encoder.layerdrop

        self.indentity = nn.Identity()

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = encoder.embed_scale
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = encoder.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_images = ImageEmbedding(embed_dim, embed_dim, image_model_name, num_image_tokens=num_image_tokens)
        self.embed_positions = encoder.embed_positions

        self.layers = encoder.layers
        self.layernorm_embedding = encoder.layernorm_embedding
        # mbart has one extra layer_norm
        self.layer_norm = encoder.layer_norm
        self.num_image_tokens = num_image_tokens

        # self.aspect_linear = nn.Linear(768, 768)
        # self.aspect_relu = nn.LeakyReLU()
        # self.aspect_linear = nn.Sequential(nn.Linear(768, 768), nn.Linear(768, 768), nn.Linear(768, 768), nn.Linear(768, 768), nn.Linear(768, 768), nn.Linear(768, 768))

    def _embed_multi_modal(self, generated_aspect_prompt, generated_senti_prompt, aspects_num, input_ids, image_features):
        """embed textual and visual inputs and combine them into one embedding"""
        # import ipdb; ipdb.set_trace()
        device = generated_aspect_prompt.device
        batch_size = input_ids.size(0)
        mask = (input_ids == self.img_feat_id) | (
            input_ids == self.cls_token_id)
        # print(mask.shape)
        embedded_images = self.embed_images(image_features)
        embedded = self.embed_tokens(input_ids)
        # print('mask shape', mask.shape)
        if not embedded_images[0].dtype == torch.float32:
            embedded = embedded.half()

        for index, value in enumerate(embedded_images):
            if len(value) > 0:
                embedded[index, mask[index]] = value
        

        new_input_ids =[]
        for i in range(len(aspects_num)):
            
            aspect_num = aspects_num[i]
            # print('the aspect_num is {}'.format(aspect_num))
           
            input_id = input_ids[i]

            if self.num_image_tokens==0:
                prompt_begin_index = 25
                prompt_end_index = 54
            elif self.num_image_tokens==1:
                prompt_begin_index = 26
                prompt_end_index = 55
            elif self.num_image_tokens==2:
                prompt_begin_index = 27
                prompt_end_index = 56
            elif self.num_image_tokens==3:
                prompt_begin_index = 28
                prompt_end_index = 57
            elif self.num_image_tokens==4:
                prompt_begin_index = 29
                prompt_end_index = 58
            elif self.num_image_tokens==5:
                prompt_begin_index = 30
                prompt_end_index = 59
            elif self.num_image_tokens==6:
                prompt_begin_index = 31
                prompt_end_index = 60
            elif self.num_image_tokens==7:
                prompt_begin_index = 32
                prompt_end_index = 61 
                
            # print('before')
            # print(len(input_id))
            # import ipdb; ipdb.set_trace()
            reserve_aspect_id = input_id[prompt_begin_index:prompt_begin_index+6*aspect_num]
            if aspect_num ==5:
                # print('aspect_num is 5')
                # print(reserve_aspect_id)
                new_input_id = torch.cat([input_id[:prompt_begin_index], reserve_aspect_id, input_id[prompt_end_index+1:]])
            else:
                cut_aspect_id = torch.ones_like(input_id[prompt_begin_index+6*aspect_num:prompt_end_index])
                new_input_id = torch.cat([input_id[:prompt_begin_index], reserve_aspect_id, cut_aspect_id, input_id[prompt_end_index:]])
            # print("++++++++++++++++++++cut_aspect_id++++++++++++++++++++++++")
            # print(cut_aspect_id)
            # print(input_id[58:])
            new_input_ids.append(new_input_id)
            # print('the shape of new_input_id is {}'.format(new_input_id.shape))
            # print(new_input_id[58:])

        new_input_ids = torch.stack(new_input_ids)

        
        if self.use_generated_aspect_prompt:
            ##aspect_prompt

            # import ipdb; ipdb.set_trace()
            aspect_prompt_mask = (new_input_ids == self.aspect_prompt_token_id) ##[29:58]: 一共5组:[50288, 50288,     9, 50289,  5702, 50284,]
            if self.use_different_aspect_prompt:
                # self.aspect_linear = self.aspect_linear.to(device)
                # self.aspect_relu = self.aspect_relu.to(device)
                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]
                    # prompt_embedding_ = generated_prompt[index].repeat(aspect_num, 1)
                    aspect_prompt_embedding_list = []
                    for j in range(aspect_num):
                        aspect_linear = nn.Linear(768, 768).to(generated_aspect_prompt.device) ##每个aspect有自己的变换，为每个aspect设计特定的prompt
                        aspect_relu = nn.LeakyReLU().to(generated_aspect_prompt.device)
                        aspect_prompt_embedding = aspect_linear(generated_aspect_prompt[index])
                        aspect_prompt_embedding = aspect_relu(aspect_prompt_embedding)
                        aspect_prompt_embedding_list.append(aspect_prompt_embedding)
                    aspect_prompt_embedding_ = torch.cat(aspect_prompt_embedding_list, dim=0)
                    embedded[index, aspect_prompt_mask[index]] = aspect_prompt_embedding_
            else:
                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]
                    aspect_prompt_embedding_ = generated_aspect_prompt[index].repeat(aspect_num, 1)
                    embedded[index, aspect_prompt_mask[index]] = aspect_prompt_embedding_

        ##sentiment_prompt
       
        if self.use_generated_senti_prompt:
            '''
            # if self.use_different_senti_prompt:
            以下使用的是attention机制，senti_prompt_token和sentiments_embdedding
            # sentiments_ids = torch.tensor([self.neu_id, self.pos_id, self.neg_id]).to(device)
            # sentiments_embdedding = self.embed_tokens(sentiments_ids)
            # senti_prompt_mask = (input_ids == self.senti_prompt_token_id)
            # for index in range(len(aspects_num)):
                
            #     aspect_num = aspects_num[index]
            #     expanded_sentiments_embdedding = sentiments_embdedding.expand(aspect_num, sentiments_embdedding.size(0), sentiments_embdedding.size(1))
            #     original_senti_prompt = embedded[index, senti_prompt_mask[index]].unsqueeze(1)
            #     new_senti_prompt = self.attention_for_senti_prompt(original_senti_prompt, expanded_sentiments_embdedding, expanded_sentiments_embdedding).squeeze()
            #     # import ipdb; ipdb.set_trace()
            #     embedded[index, senti_prompt_mask[index]] = new_senti_prompt
            '''
            ##换成senti_prompt也是生成形式看看
            senti_prompt_mask = (new_input_ids == self.senti_prompt_token_id)
            # import ipdb; ipdb.set_trace()
            if self.use_different_senti_prompt:
                # self.aspect_linear = self.aspect_linear.to(generated_senti_prompt.device)
                # self.aspect_relu = self.aspect_relu.to(generated_senti_prompt.device)
            
                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]
                    # prompt_embedding_ = generated_prompt[index].repeat(aspect_num, 1)
                    prompt_embedding_list = []
                    for j in range(aspect_num):
                        senti_linear = nn.Linear(768, 768).to(generated_senti_prompt.device)
                        senti_relu = nn.LeakyReLU().to(generated_senti_prompt.device)
                        prompt_embedding = senti_linear(generated_senti_prompt[index])
                        prompt_embedding = senti_relu(prompt_embedding)
                        prompt_embedding_list.append(prompt_embedding)
                    prompt_embedding_ = torch.cat(prompt_embedding_list, dim=0)
                    embedded[index, senti_prompt_mask[index]] = prompt_embedding_
            else:
                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]
                    prompt_embedding_ = generated_senti_prompt[index].repeat(aspect_num, 1)

                    embedded[index, senti_prompt_mask[index]] = prompt_embedding_


        return embedded
       

    def forward(self,
                input_ids,
                image_features,
                attention_mask=None,
                generated_aspect_prompt=None,
                generated_senti_prompt=None,
                aspects_num=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param output_attentions:
        :param output_hidden_states:
        :return: Tuple comprised of:
            - x (Tensor): the last encoder layer's output of
              shape (src_len, batch, embed_dim)
            - encoder_states (List[Tensor]): all intermediate
              hidden states of shape (src_len, batch, embed_dim).
              Only populated if output_hidden_states: is True.
            - all_attentions (List[Tensor]): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self._embed_multi_modal(generated_aspect_prompt, generated_senti_prompt, aspects_num,
            input_ids, image_features) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability <
                                  self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x,
                                        attention_mask,
                                        output_attentions=output_attentions)

            if output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [
            hidden_state.transpose(0, 1) for hidden_state in encoder_states
        ]
        x = x.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions]
                         if v is not None)
        return BaseModelOutput(last_hidden_state=x,
                               hidden_states=encoder_states,
                               attentions=all_attentions)

class MultiModalBartDecoder_span(nn.Module
                                 ):  #AOE task and all downstream tasks
    def __init__(self,
                 config: MultiModalBartConfig,
                 tokenizer,
                 decoder,
                 pad_token_id,
                 label_ids,
                 causal_mask,
                 num_image_tokens=2,
                 need_tag=True,
                 only_sc=False,
                 avg_feature=False,
                 use_encoder_mlp=True):
        super().__init__()
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.causal_mask = causal_mask
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        # label_ids = sorted(label_ids, reverse=False)
        self.label_start_id = min(label_ids)
        self.label_end_id = max(label_ids) + 1
        self.need_tag = need_tag
        self.only_sc = only_sc
        self.num_image_tokens = num_image_tokens
        mapping = torch.LongTensor([0, 2] + label_ids)
        ###mapping: [0, 2, 50276, 50277, 50278, 50281]
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)  # 加上一个
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.dropout_layer = nn.Dropout(0.1)

        self.end_text_id = tokenizer.end_text_id
        self.avg_feature = avg_feature
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.Dropout(0.3),
                nn.ReLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, tokens, state, only_sc=False):
        # import ipdb; ipdb.set_trace()
        '''
        tokens: [[0, 2, 2, 16, 16, 4, 18, 18, 4, 1, 1, 1, 1],
                 [0, 2, 2, 15, 16, 3, 25, 26, 5, 28, 28, 4, 1]]
        '''
        # import ipdb; ipdb.set_trace()
        bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output ##(batch, 72=38(len(image_token+begin_image+end_image(36+1+1)))+34(max_tex_len(包含begin_text_id(0) and end_text_id(2)) in batch), 768)
        encoder_pad_mask = state.encoder_mask ##(batch, 72)
        first = state.first
        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(
            self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index  # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(
            src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens 
        # print(src_tokens.shape): (2, 34)
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1) ###Sequence
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)
        # print('word_mapped_tokens', word_mapped_tokens)
        tokens = torch.where(mapping_token_mask, tag_mapped_tokens,
                             word_mapped_tokens)

        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)
        '''
        {'AESC': 50281, 'POS': 50276, 'NEU': 50277, 'NEG': 50278}
        tensor([[0, 50276, 50276, 4644, 4644, 50278, 798, 798, 50278, 2, 1, 1, 1],
                [0, 50276, 50276, 9517, 957, 50277, 2561, 7772, 50281, 2762, 2762, 50278, 2]])
        将tokens中的index以及标签都转化为vocabulary中的token_id
        '''

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(
                self.pad_token_id)  # decoder需要让pad位置为1

            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.
                                causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:

            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=self.
                                causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)

        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size (2, 12(去掉了 end_token_id), 768)
        hidden_state = self.dropout_layer(hidden_state)
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full(
            (hidden_state.size(0), hidden_state.size(1),
             self.src_start_index + src_tokens.size(-1)),
            fill_value=-1e24)
        ##建立空的logits
        # print('logits', logits.shape) (bsz, max_len,  self.src_start_index + src_tokens.size(-1)) -> (2, 12, 40=6+34)
        # 首先计算的是

        if self.need_tag:
            '''
            self.decoder.embed_tokens.weight: (50289, 768)
            self.label_start_id: 50276
            '''
            tag_scores = F.linear(
                hidden_state,
                self.dropout_layer(
                    self.decoder.embed_tokens.
                    weight[self.label_start_id:self.label_start_id +
                           3]))  # bsz x max_len x num_class
            logits[:, :, 3:self.src_start_index] = tag_scores ###给情感的position赋值[:, :, (3, 4, 5)]
        if not only_sc:
            eos_scores = F.linear(
                hidden_state,
                self.dropout_layer(self.decoder.embed_tokens.weight[2:3])) 
            '''
            ['</s>(eos_token)', '<mask>', '<pad>', '<s>(bos_token)', '<unk>']
            [2, 50264, 1, 0, 3]
            '''

            # bsz x max_bpe_len(image_len + text_len) x hidden_size: (2, 72, 768)
            src_outputs = state.encoder_output 
            if self.num_image_tokens==0:
                end_index = 62
            elif self.num_image_tokens==1:
                end_index = 63
            elif self.num_image_tokens==2:
                end_index = 64
            elif self.num_image_tokens==3:
                end_index = 65
            elif self.num_image_tokens==4:
                end_index = 66
            elif self.num_image_tokens==5:
                end_index = 67
            elif self.num_image_tokens==6:
                end_index = 68
            elif self.num_image_tokens==7:
                end_index = 69

            
            if hasattr(self, 'encoder_mlp') and not only_sc:
                src_outputs = self.encoder_mlp(src_outputs)

            if first is not None:
                mask = first.eq(0)
                src_outputs = src_outputs.gather(
                    index=first.unsqueeze(2).repeat(1, 1,
                                                    src_outputs.size(-1)),
                    dim=1)
            else:
                mask = state.encoder_mask[:, end_index:].eq(0)
                # src_outputs = self.decoder.embed_tokens(src_tokens)
            mask = mask.unsqueeze(1) ## bsz x 1 x max_word_len: (2, 1, 34)
            input_embed = self.decoder.embed_tokens(
                src_tokens)  #bsz x max_word_len x hidden_size: (2, 34, 768); src_tokens: (2, 34)
            input_embed = self.dropout_layer(input_embed)
            if self.avg_feature:  # 先把feature合并一下
                src_outputs = (src_outputs[:, end_index:] + input_embed) / 2
            word_scores = torch.einsum(
                'blh,bnh->bln', hidden_state,
                src_outputs[:, end_index:])  # bsz x max_len x max_word_len: (2, 12, 34)
            if not self.avg_feature:
                gen_scores = torch.einsum(
                    'blh,bnh->bln', hidden_state,
                    input_embed)  # bsz x max_len x max_word_len: (2, 12, 34)
                word_scores = (gen_scores + word_scores) / 2 
            mask = mask.__or__(
                src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1)) ###(2, 1, 34)
            word_scores = word_scores.masked_fill(mask, -1e32) ###(bts, max_len, max_word_len)
            logits[:, :, self.src_start_index:] = word_scores
            ###logits.shape (bts, max_len, max_word_len+6): (2, 12, 40)
            logits[:, :, 1:2] = eos_scores
        # print(torch.argmax(logits[0], dim=-1))
        return logits

    def decode(self, tokens, state, only_sc=False):
        return self(tokens, state, only_sc)[:, -1]



class Span_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.loss_fct = nn.CrossEntropyLoss()
        self.fc = nn.LogSoftmax(dim=-1)

    def forward(self, tgt_tokens, pred, mask):
        '''
        tgt_tokens: (2 (batch-size), 12 (max_len+1))
        pred: (2, 12, 40 (max_word_len))
        '''

        tgt_tokens = tgt_tokens.masked_fill(mask.eq(0), -100)
        output = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2)) ##每一个词都有12种类别， input= (40, 12)
        return output


class MultiModalBartDecoder_MLM(nn.Module):
    def __init__(self, config: MultiModalBartConfig, decoder):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.register_buffer(
            "final_logits_bias",
            torch.zeros((1, self.decoder.embed_tokens.num_embeddings)))

    def forward(self, labels, input_ids, encoder_outputs, attention_mask,
                decoder_input_ids, decoder_attention_mask):
        decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
            self.config,
            input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_attention_mask,
            causal_mask_dtype=self.decoder.embed_tokens.weight.dtype)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs,
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask[:decoder_input_ids.size(1), :
                                            decoder_input_ids.size(1)],
        )

        lm_logits = F.linear(decoder_outputs[0][:, 1:],
                             self.decoder.embed_tokens.weight,
                             bias=self.final_logits_bias)

        lm_loss = 0
        # compute lm loss if labels is given
        if labels is not None:
            labels = labels.clone()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                lm_logits.view(-1, self.decoder.embed_tokens.weight.size(0)),
                labels.reshape(-1))

            return lm_loss


class MultiModalBartDecoder_ANP_generate(nn.Module):  #AOG task
    def __init__(self, config: MultiModalBartConfig, decoder):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.register_buffer(
            "final_logits_bias",
            torch.zeros((1, self.decoder.embed_tokens.num_embeddings)))

    def forward(self, labels, input_ids, encoder_outputs, attention_mask,
                decoder_input_ids, decoder_attention_mask):
        decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
            self.config,
            input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_attention_mask,
            causal_mask_dtype=self.decoder.embed_tokens.weight.dtype)

        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs,
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask[:decoder_input_ids.size(1), :
                                            decoder_input_ids.size(1)],
        )

        lm_logits = F.linear(decoder_outputs[0][:, 1:],
                             self.decoder.embed_tokens.weight,
                             bias=self.final_logits_bias)

        lm_loss = 0
        # compute lm loss if labels is given
        if labels is not None:
            labels = labels.clone()
            # labels[labels == self.cls_token_id] = -100
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                lm_logits.view(-1, self.decoder.embed_tokens.weight.size(0)),
                labels.reshape(-1))

            return lm_loss


class MultiModalBartDecoder_sentiment(nn.Module):  #MSP task
    def __init__(self,
                 config: MultiModalBartConfig,
                 decoder,
                 senti_ids,
                 senti_nums=3):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.senti_ids = senti_ids
        self.dropout_layer = nn.Dropout(0.1)
        self.senti_head = BartClassificationHead(config.d_model,
                                                 config.d_model, senti_nums,
                                                 config.classif_dropout)

    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, senti_labels, encoder_outputs, attention_mask,
                senti_decoder_input_ids):

        decoder_outputs = self.decoder(
            input_ids=senti_decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_padding_mask=attention_mask,
            decoder_padding_mask=None,
            decoder_causal_mask=None,
        )

        # predict_senti = F.linear(
        #     decoder_outputs[0][:, 1],
        #     self.dropout_layer(self.decoder.embed_tokens.
        #                        weight[self.senti_ids[0]:self.senti_ids[2] +
        #                               1]))  # bsz
        # predict_senti = torch.flip(predict_senti, dims=[-1])
        predict_senti = self.senti_head(decoder_outputs[0][:, 1])
        loss_fct = nn.CrossEntropyLoss()
        senti_loss = loss_fct(predict_senti, senti_labels)
        return senti_loss, predict_senti


class MultiModalBartDecoder_MRM(nn.Module):
    def __init__(self, config: MultiModalBartConfig, decoder, causal_mask,
                 args):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.causal_mask = causal_mask
        self.args = args
        self.mrm_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classif_dropout,
        )
        self._init_weights(self.mrm_head.dense)
        self._init_weights(self.mrm_head.out_proj)

    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, mrm_labels, mrm_masks, encoder_outputs, attention_mask,
                mrm_decoder_input_ids, mrm_decoder_attention_mask):

        decoder_padding_mask = mrm_decoder_attention_mask.eq(0)
        decoder_outputs = self.decoder(
            input_ids=mrm_decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_padding_mask=attention_mask,
            decoder_padding_mask=decoder_padding_mask,
            decoder_causal_mask=self.causal_mask[:mrm_decoder_input_ids.size(
                1), :mrm_decoder_input_ids.size(1)].to(
                    mrm_decoder_input_ids.device),
        )
        region_representation = decoder_outputs[0][mrm_masks.bool()]
        if len(region_representation) > 0:
            predict_cls = self.mrm_head(region_representation)
            loss_fct = nn.CrossEntropyLoss()
            mrm_labels = torch.cat(mrm_labels,
                                   dim=0).to(encoder_outputs.device)

            if self.args.mrm_loss_type == 'KL':
                predict_cls = F.log_softmax(predict_cls, dim=-1)
                mrm_loss = F.kl_div(predict_cls.double(),
                                    mrm_labels.double().squeeze(1),
                                    reduction='batchmean')
            else:
                raise RuntimeError("wrong mrm type")
        else:
            mrm_loss = 0

        return mrm_loss


'''
generate_aspect_prompt based on the multimodal context
'''
class MultiModalBartDecoder_generate_aspect_prompt(nn.Module): 
    def __init__(self, config: MultiModalBartConfig, decoder):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.aspect_prompt_linear = nn.Linear(768, 768)


    def forward(self, encoder_outputs, attention_mask,
                decoder_input_ids, decoder_attention_mask):

        # import ipdb; ipdb.set_trace()
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_padding_mask=attention_mask.eq(0),
            decoder_padding_mask=decoder_attention_mask.eq(0),
            decoder_causal_mask=None,
        )

        prompt_logits = decoder_outputs[0]
        aspect_prompt_logits = self.aspect_prompt_linear(prompt_logits)


        return aspect_prompt_logits 


'''
generate_sentiment_prompt based on the multimodal context
'''
class MultiModalBartDecoder_generate_sentiment_prompt(nn.Module): 
    def __init__(self, config: MultiModalBartConfig, decoder):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.senti_prompt_linear = nn.Linear(768, 768)


    def forward(self, encoder_outputs, attention_mask,
                decoder_input_ids, decoder_attention_mask):

        # import ipdb; ipdb.set_trace()
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_padding_mask=attention_mask.eq(0),
            decoder_padding_mask=decoder_attention_mask.eq(0),
            decoder_causal_mask=None,
        )

        prompt_logits = decoder_outputs[0]
        senti_prompt_logits = self.senti_prompt_linear(prompt_logits)


        return senti_prompt_logits


class MultiModalBartDecoder_aspects_num(nn.Module):  #MSP task
    def __init__(self,
                 config: MultiModalBartConfig,
                 decoder,
                 max_aspects_nums=5):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.dropout_layer = nn.Dropout(0.1)
        self.aspects_num_head = BartClassificationHead(config.d_model,
                                                 config.d_model, max_aspects_nums,
                                                 config.classif_dropout)

    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, aspects_num_labels, encoder_outputs, attention_mask,
                aspects_num_decoder_input_ids):
        
        decoder_outputs = self.decoder(
            input_ids=aspects_num_decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_padding_mask=attention_mask,
            decoder_padding_mask=None,
            decoder_causal_mask=None,
        )

        # predict_aspects_num = F.linear(
        #     decoder_outputs[0][:, 1],
        #     self.dropout_layer(self.decoder.embed_tokens.
        #                        weight[self.aspects_num_ids[0]:self.aspects_num_ids[2] +
        #                               1]))  # bsz
        # predict_aspects_num = torch.flip(predict_aspects_num, dims=[-1])
        predict_aspects_num_logits = self.aspects_num_head(decoder_outputs[0][:, 1])
        loss_fct = nn.CrossEntropyLoss()
        aspects_num_labels = torch.tensor(aspects_num_labels).to(predict_aspects_num_logits.device)
        aspects_num_loss = loss_fct(predict_aspects_num_logits, aspects_num_labels)
        return aspects_num_loss, predict_aspects_num_logits