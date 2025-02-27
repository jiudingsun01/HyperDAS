from torch.utils.data import DataLoader
import os
import json
from datasets import load_from_disk
from hyperdas.data.axbench import get_ravel_collate_fn
import argparse


from transformers import AutoTokenizer


def run_experiment(
    inference_modes=["default", "bidding_argmax"],
    intervention_layer=15,
    subspace_module="ReflectSelect",
    model_name_or_path="./models/llama3-8b",
    checkpoint_path=None,
    batch_size=16,
    source_suffix_visibility=False,
    base_suffix_visibility=False,
    save_dir=None,
    das_dimension=128,
    test_path=None,
):
    if "default" in inference_modes:
        inference_modes.remove("default")
        inference_modes.append(None)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    test_set = load_from_disk(test_path)

    collate_fn = get_ravel_collate_fn(
        tokenizer,
        source_suffix_visibility=source_suffix_visibility,
        base_suffix_visibility=base_suffix_visibility,
        add_space_before_target=True,
    )
    test_data_loader = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )

    from src.hyperdas.llama3.model import RavelInterpretor

    hypernetwork = RavelInterpretor(
        model_name_or_path=model_name_or_path,
        num_editing_heads=32,
        intervention_layer=intervention_layer,
        subspace_module=subspace_module,
        das_dimension=das_dimension,
    )

    hypernetwork = hypernetwork.to("cuda")
    hypernetwork.load_model(checkpoint_path)

    result_dict = {}
    for inference_mode in inference_modes:
        accs, test_loss, correct_indices = hypernetwork.predict(
            test_data_loader, inference_mode=inference_mode
        )
        if inference_mode is None:
            inference_mode = "default"
        result_dict[inference_mode] = {
            "accs": accs,
            "test_loss": test_loss,
            "correct_indices": correct_indices,
        }

        for k, v in accs.items():
            print(f"{inference_mode} {k}: {v}")

    if save_dir is not None:
        with open(os.path.join(save_dir), "w") as f:
            json.dump(result_dict, f)
        print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervention_layer", type=int, default=15)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--model_name_or_path", type=str, default="/nlp/scr/sjd24/llama3-8b"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--source_suffix_visibility", default=False, action="store_true"
    )
    parser.add_argument("--base_suffix_visibility", default=False, action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument(
        "--test_path", type=str, default="./experiments/ravel/data/city_country_test"
    )

    parser.add_argument(
        "--inference_modes",
        nargs="+",
        default=["default", "column_argmax", "bidding_argmax"],
    )

    # if None, use Boundless DAS
    parser.add_argument(
        "--subspace_module",
        default="ReflectSelect",
        choices=[None, "DAS", "BoundlessDAS", "MaskSelect", "ReflectSelect"],
    )
    parser.add_argument("--das_dimension", type=int, default=128)

    args = parser.parse_args()
    args = dict(args.__dict__)
    run_experiment(**args)
