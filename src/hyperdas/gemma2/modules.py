import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Config,
    Gemma2Model,
    Gemma2ForCausalLM,
)

from typing import Any, List, Mapping, Optional, Tuple, TypeVar, Union

from transformers.cache_utils import StaticCache, DynamicCache, Cache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..das_utils import BoundlessRotatedSpaceIntervention, RotatedSpaceIntervention, LowRankRotatedSpaceIntervention, SelectiveLowRankRotatedSpaceIntervention,ReflectiveLowRankRotatedSpaceIntervention
from ..utils import (
    InterpretorModelOutput,
    add_fwd_hooks,
    assign_layer_indices,
)

T = TypeVar("T", bound="Gemma2Interpretor")


class Gemma2InterpretorConfig(Gemma2Config):
    boundless_das: bool = False
    torch_dtype = torch.bfloat16
    chop_editor_at_layer: int = -1
    num_editing_heads: int = 32
    compute_position_ids: bool = True
    intervention_layer: int = 24
    initialize_from_scratch: bool = False
    ablate_base_token_attention: bool = False
    ablate_source_token_attention: bool = False
    break_asymmetric: bool = False


class Gemma2ModelWithCrossAttention(Gemma2Model):
    pass


class Gemma2InterpretorHypernetwork(Gemma2ForCausalLM):
    def __init__(self, config: Gemma2InterpretorConfig):
        super().__init__(config)
        


class Gemma2Interpretor(nn.Module):
    def __init__(self, config: Gemma2InterpretorConfig, subspace_module=None, das_dimension=None):
        super().__init__()

        self.config = config
        # self.hypernetwork = Gemma2InterpretorHypernetwork(config)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            config.name_or_path, torch_dtype = config.torch_dtype
        )
        
        self.bidding_threshold = 0.1
        
        self.use_das_intervention = subspace_module != None
        self.das_selective_subspace = subspace_module in ["ReflectSelect", "MaskSelect"]
                
        if self.use_das_intervention:
            
            if subspace_module == "BoundlessDAS":
                self.das_module = BoundlessRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size, torch_dtype=config.torch_dtype
                )
            elif subspace_module == "DAS":
                self.das_module = LowRankRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size, low_rank_dimension=das_dimension, torch_dtype=config.torch_dtype
                )
            elif subspace_module == "MaskSelect":
                self.das_module = SelectiveLowRankRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size, low_rank_dimension=das_dimension, torch_dtype=config.torch_dtype
                )
            elif subspace_module == "ReflectSelect":
                self.das_module = ReflectiveLowRankRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size, low_rank_dimension=das_dimension, torch_dtype=config.torch_dtype
                )
            else:
                raise ValueError("Invalid subspace module")
                    
        # freeze target model
        for param in self.target_model.parameters():
            param.requires_grad = False

        assign_layer_indices(self.target_model)
        
        self.layerwise_embeddings = None

    def train(self: T, mode: bool = True) -> T:
        return self.hypernetwork.train(mode)

    def eval(self: T) -> T:
        return self.hypernetwork.eval()

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """Only load weights for the trainable hypernetwork."""
        self.hypernetwork.load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def _run_target_model_for_encoded_hidden_states(
        self,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        position_ids: torch.Tensor = None,
    ):
        """Gets the hidden states from the target model, if necessary"""

        if position_ids is not None:
            outputs = self.target_model(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )

        else:
            outputs = self.target_model(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
                output_hidden_states=True,
            )

        return outputs.hidden_states

    def forward(
        self,
        editor_input_ids: torch.Tensor = None,
        editor_attention_mask: torch.Tensor = None,
        base_input_ids: torch.Tensor = None,
        base_attention_mask: torch.Tensor = None,
        base_intervention_mask: torch.Tensor = None,
        source_input_ids: torch.Tensor = None,
        source_attention_mask: torch.Tensor = None,
        source_intervention_mask: torch.Tensor = None,
        base_hidden_states: torch.Tensor = None,
        base_position_ids: torch.Tensor = None,
        source_hidden_states: torch.Tensor = None,
        source_position_ids: torch.Tensor = None,
        intervention_layer: int = None,
        output_vanilla_hidden_states: bool = True,
        output_edited_hidden_states: bool = False,
        output_intervention_weight: bool = True,
        intervention_weight: torch.Tensor = None,
        inference_mode: str = None,
    ) -> InterpretorModelOutput:
        
        assert inference_mode in [None, "column_argmax", "global_argmax", "groundtruth", "bidding_argmax"]
        
                
        if intervention_layer is None:
            intervention_layer = self.config.intervention_layer 
            
        if base_position_ids is None:
            # -1 for all the padding tokens and start from 0 for the rest
            base_position_ids = torch.cumsum(base_attention_mask, dim=1) * base_attention_mask - 1
            
        if source_position_ids is None:
            # -1 for all the padding tokens and start from 0 for the rest
            source_position_ids = torch.cumsum(source_attention_mask, dim=1) * source_attention_mask - 1
        
        # Run target model for encoded hidden states
        if base_hidden_states is None:
            base_hidden_states = torch.stack(
                self._run_target_model_for_encoded_hidden_states(
                    base_input_ids, base_attention_mask, base_position_ids
                ),  # seems to break while we are passing thru batch_size=1; the last (12th =) has different dimensions
                dim=2,
            )
        
        if source_hidden_states is None:
            source_hidden_states = torch.stack(
                self._run_target_model_for_encoded_hidden_states(
                    source_input_ids, source_attention_mask, source_position_ids
                ),
                dim=2,
            )
            
        if base_intervention_mask is None:
            if base_attention_mask is not None:
                base_intervention_mask = base_attention_mask.clone()
            else:
                base_intervention_mask = torch.ones_like(base_input_ids)
                
        if source_intervention_mask is None:
            if source_attention_mask is not None:
                source_intervention_mask = source_attention_mask.clone()
            else:
                source_intervention_mask = torch.ones_like(source_input_ids)
            
        # dimensions of target_hidden_states:
        # batch_size, token_sequence_length, num_layers = 13, resid_width = 768
        # Normalize along the last dimension
        base_normalization_factors = base_hidden_states.norm(dim=-1, keepdim=True)
        base_hidden_states = base_hidden_states / base_normalization_factors
        
        source_normalization_factors = source_hidden_states.norm(dim=-1, keepdim=True)
        source_hidden_states = source_hidden_states / source_normalization_factors
        
        print("base_hidden_states.shape", base_hidden_states.shape)
        print("source_hidden_states.shape", source_hidden_states.shape)
        
        
        raise

        if intervention_weight is None or inference_mode == "groundtruth":
            
            n_layer = base_hidden_states.shape[2]
                        
            collapsed_base_hidden_states = base_hidden_states.reshape(
                base_hidden_states.shape[0],
                base_hidden_states.shape[1] * base_hidden_states.shape[2],
                base_hidden_states.shape[3],
            )
            
            collapsed_base_attention_mask = base_intervention_mask.unsqueeze(-1).repeat(1, 1, n_layer)
            collapsed_base_attention_mask = collapsed_base_attention_mask.reshape(
                base_intervention_mask.shape[0],
                base_intervention_mask.shape[1] * n_layer,
            )
            
            collapsed_source_hidden_states = source_hidden_states.reshape(
                source_hidden_states.shape[0],
                source_hidden_states.shape[1] * source_hidden_states.shape[2],
                source_hidden_states.shape[3],
            )
            
            collapsed_source_attention_mask = source_intervention_mask.unsqueeze(-1).repeat(1, 1, n_layer)
            collapsed_source_attention_mask = collapsed_source_attention_mask.reshape(
                source_intervention_mask.shape[0],
                source_intervention_mask.shape[1] * n_layer,
            )

            interpretor_output = self.hypernetwork(
                input_ids=editor_input_ids,
                attention_mask=editor_attention_mask,
                base_hidden_states=collapsed_base_hidden_states,
                base_attention_mask=collapsed_base_attention_mask,
                source_hidden_states=collapsed_source_hidden_states,
                source_attention_mask=collapsed_source_attention_mask,
                use_cache=False
            )
            
            if inference_mode == "groundtruth":
                hypernet_hidden_states, _ = interpretor_output
                intervention_weight = intervention_weight.to(dtype=hypernet_hidden_states.dtype)
            else:
                # Multiply the outputs by normalization factors
                hypernet_hidden_states, intervention_weight = interpretor_output
                intervention_weight = intervention_weight.squeeze()
            
        if inference_mode == "global_argmax":
            batch_size, _, num_base_pos = intervention_weight.shape
            source_base_intervention_flatten = intervention_weight[:, :-1, :].view(batch_size, -1)
            max_intervention_position = torch.argmax(source_base_intervention_flatten, dim=1)
            intervention_weight = torch.zeros_like(intervention_weight)
            intervention_weight[:, -1, :] = 1.0
            for i in range(batch_size):
                source_token_idx = max_intervention_position[i] // num_base_pos
                base_token_idx = max_intervention_position[i] % num_base_pos
                intervention_weight[i, source_token_idx, base_token_idx] = 1.0
                intervention_weight[i, -1, base_token_idx] = 0.0
        elif inference_mode == "column_argmax":
            batch_size, num_src_pos, num_base_pos = intervention_weight.shape
            intervention_weight = torch.argmax(intervention_weight, dim=1)
            intervention_weight = torch.nn.functional.one_hot(intervention_weight, num_classes=num_src_pos).to(dtype=intervention_weight.dtype).permute(0, 2, 1)            
        elif inference_mode == "bidding_argmax":
            batch_size, num_src_pos, num_base_pos = intervention_weight.shape
            bidding_weight = torch.argmax(intervention_weight[:, :-1, :], dim=-1)
            bidding_weight = torch.nn.functional.one_hot(bidding_weight, num_classes=num_base_pos).float()
            bidding_weight = torch.cat([bidding_weight, torch.ones(batch_size, 1, num_base_pos).to(bidding_weight.device)], dim=1)
            intervention_weight = torch.where(bidding_weight == 1, intervention_weight, torch.zeros_like(intervention_weight))
            if self.bidding_threshold is not None:
                threshold = torch.Tensor([self.bidding_threshold]).to(intervention_weight.device)
                threshold = threshold.repeat(batch_size, num_base_pos)
                intervention_weight[:, -1, :] = torch.where(intervention_weight[:, -1, :] > self.bidding_threshold, intervention_weight[:, -1, :], threshold)
            intervention_weight = torch.argmax(intervention_weight, dim=1)
            intervention_weight = torch.nn.functional.one_hot(intervention_weight, num_classes=num_src_pos).to(dtype=intervention_weight.dtype).permute(0, 2, 1)
    

        if len(intervention_weight.shape) == 2:
            intervention_weight = intervention_weight.unsqueeze(0) # Unsqueeze first dim if batch size = 1
                
        source_output = self.target_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            output_hidden_states=True,
        )
        
        source_hidden_states = source_output.hidden_states[intervention_layer]
        
        intervention_matrix = torch.einsum("bij,bid->bijd", intervention_weight[:, :-1, :], source_hidden_states) # TODO: Fix it to help the new implement 
        intervention_matrix = intervention_matrix.sum(dim=1) 
        
        # Run target model with edit vectors.
        # This adds the edit vectors to the given hidden state at the specified batch index, position, and layer
        def representation_swap(module, input, output):
            base_hidden_states = output[0].clone()
            batch_size = base_hidden_states.shape[0]
            base_intervention_weight = intervention_weight[:, -1, :]
            
            if self.use_das_intervention:
                
                source_intervention_hidden_states = intervention_matrix + torch.einsum("bid,bi->bid", base_hidden_states, base_intervention_weight)
                
                if self.das_selective_subspace:
                    mixed_output = self.das_module(base_hidden_states, source_intervention_hidden_states, hypernet_hidden_states)
                else:
                    mixed_output = self.das_module(base_hidden_states, source_intervention_hidden_states, batch_size)
                
                output[0][:] += (mixed_output - base_hidden_states)
            else:
                res_diff = torch.einsum("bid,bi->bid", base_hidden_states, (1 - base_intervention_weight))
                output[0][:] += (intervention_matrix - res_diff)
            
        def embedding_representation_swap(module, input, output):
            if self.use_das_intervention:
                raise NotImplementedError("DAS intervention is not supported for token embeddings")
            
            base_hidden_states = output.clone()
            base_intervention_weight = intervention_weight[:, -1, :]
            res_diff = torch.einsum("bid,bi->bid", base_hidden_states, (1 - base_intervention_weight))
            output += (intervention_matrix - res_diff)    
        
        # Now editing the target model
        if intervention_layer == 0:
            hooks = [(self.target_model.model.embed_tokens, embedding_representation_swap)]
        else:
            hooks = [(self.target_model.model.layers[intervention_layer - 1], representation_swap)]

        with add_fwd_hooks(hooks):
            # THIS IS THE LINE WHERE THE MODEL IS CALLED (AND THE EDITOR IS CALLED AT
            # THE END OF `layer` AS A SIDE EFFECT)
            target_result = self.target_model(
                input_ids=base_input_ids,
                attention_mask=base_attention_mask,
                position_ids=base_position_ids,
                output_hidden_states=output_edited_hidden_states,
            )
    
        logits = target_result.logits
        
        output = InterpretorModelOutput(logits=logits)
        if output_edited_hidden_states:
            output.edited_hidden_states = target_result.hidden_states
            
        if output_intervention_weight:
            output.intervention_weight = intervention_weight
            
        if output_vanilla_hidden_states:
            output.vanilla_base_hidden_states = base_hidden_states
            output.vanilla_source_hidden_states = source_hidden_states

        return output