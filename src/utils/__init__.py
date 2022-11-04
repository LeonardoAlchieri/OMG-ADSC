from typing import Union
from torch import load as load_weights

def load_backbone_weight(
    weights_path: str, loading_device: str = "cuda"
) -> Union[dict, None]:
    """Simple method to load pre-trained weights for some of the backbones
    available.

    Parameters
    ----------
    weights_path : str
        path to the weights file

    Returns
    -------
    dict
        weights for the backbone
    """
    if weights_path:
        print(f"Loading pre-trained weights from {weights_path}")
        checkpoint = load_weights(weights_path, map_location=loading_device)

        if "state_dict" in checkpoint.keys():
            state_dict: dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint.keys():
            state_dict: dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            print("No state_dict or model_state_dict found in weights file")

        return state_dict
    else:
        print("No loading of pre-trained weights. Maybe already loaded?")
        return None