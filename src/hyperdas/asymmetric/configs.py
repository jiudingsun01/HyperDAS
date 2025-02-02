from transformers.models.llama.modeling_llama import LlamaConfig
import torch


class HyperDASSubspaceConfig:
    subspace_dimension: int
    
class QuasiProjectiveConfig(HyperDASSubspaceConfig):
    is_subspace_intervention: bool
    subspace_module: str
    subspace_module = "QuasiProjective"
    is_subspace_intervention = True
    dict_size: int = 32
    scoring_dimension: int = 32
    lambda_parameter: float = 0.001
    importance_power: int = -2
    epsilon=0.000001
    return_penalty: bool = True
    ridge_parameterization = None
    compute_metrics: bool = True
    orthogonal_init: bool = True
    selection_mechanism: str = "dynamic"
    hat_matrix: bool = True
    
class HouseholderConfig(HyperDASSubspaceConfig):
    subspace_module = "ReflectSelect"
    is_subspace_intervention = True

class FullRankConfig(HyperDASSubspaceConfig):
    subspace_module = "FullRank"
    is_subspace_intervention = False
    
    
class BoundlessDASConfig(HyperDASSubspaceConfig):
    subspace_module = "Boundless"
    is_subspace_intervention = True
    
class MDASConfig(HyperDASSubspaceConfig):
    subspace_module = "MDAS"
    is_subspace_intervention = True
    
    
class MaskSelectConfig(HyperDASSubspaceConfig):
    subspace_module = "MaskSelect"
    is_subspace_intervention = True
    
        
class HyperDASConfig(LlamaConfig):
    torch_dtype = torch.bfloat16
    num_decoders: int = -1
    num_editing_heads: int = 32
    compute_position_ids: bool = True
    intervention_layer: int = 24
    initialize_from_pretrained: bool = False
    target_model_num_hidden_layers: int = None
    target_model_name_or_path: str = None
    subspace_config: HyperDASSubspaceConfig = HyperDASSubspaceConfig()

        
        
class AsymmetricHyperDASConfig(HyperDASConfig):
    break_asymmetric: bool = False
    inference_modes: list = ["bidding_argmax", "default"]
    ablate_base_token_attention: bool = False
    ablate_source_token_attention: bool = False

class SymmetricHyperDASConfig(HyperDASConfig):
    pass


    
    