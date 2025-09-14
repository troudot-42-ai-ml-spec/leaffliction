import numpy as np
from plantcv import plantcv as pcv
from typing import Dict, Any
from ..registry import register


@register("analyse")
class Analyse:
    name = "analyse"

    def __init__(self) -> None: ...

    def apply(self, img: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if "mask" not in ctx:
            raise Exception("OtsuMask has to be called before Analyse!")
        masks: Dict[str, np.ndarray] = ctx["mask"]
        pcv.params.sample_label = "leaf"
        ctx["analyse"] = {}
        ctx["analyse_value"] = {}
        ctx["analyse_results"] = {}  # Store results directly in context

        for i, (channel, mask) in enumerate(masks.items(), start=1):
            label = f"{pcv.params.sample_label}_{i}"
            ctx["analyse_value"][channel] = label

            analysis_result = pcv.analyze.size(img, mask, n_labels=1)

            obs = pcv.outputs.observations

            if obs:
                current_obs_key = list(obs.keys())[-1]
                area_val = obs[current_obs_key]["area"]["value"]
                perimeter_val = obs[current_obs_key]["perimeter"]["value"]
                width_val = obs[current_obs_key]["width"]["value"]
                height_val = obs[current_obs_key]["height"]["value"]
                centroid_x = obs[current_obs_key]["center_of_mass"]["value"][0]
                centroid_y = obs[current_obs_key]["center_of_mass"]["value"][1]

                ctx["analyse_results"][channel] = {
                    "area": area_val,
                    "perimeter": perimeter_val,
                    "width": width_val,
                    "height": height_val,
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
                }
            else:
                print(f"Warning: No observations found for channel {channel}")
                ctx["analyse_results"][channel] = {
                    "area": 0,
                    "perimeter": 0,
                    "width": 0,
                    "height": 0,
                    "centroid_x": 0,
                    "centroid_y": 0,
                }

            ctx["analyse"][channel] = analysis_result
        return img
