from glob import glob
from os.path import basename
from os.path import join as join_paths
from typing import Any, Dict

from execution_time_wrapper import get_execution_time_print
from pandas import DataFrame, Series, read_csv
from yaml import safe_load as load_yaml

_filename: str = basename(__file__).split(".")[0]


@get_execution_time_print
def load_config(path: str = "repeat_src/run/config_dan_fcn.yml") -> Dict[str, Any]:
    with open(path, "r") as file:
        config_params = load_yaml(file)
    return config_params


@get_execution_time_print
def create_img_list(row: Series, data_path: str) -> str:
    video, utterance = row
    img_list = glob(join_paths(data_path, video, utterance, "*.png"))
    img_list = [el.split("/")[-1].split(".")[0] for el in img_list]
    return ",".join(img_list)


def main():
    path_to_config: str = f"utils/config_{_filename}.yml"

    print("Starting model training")
    configs = load_config(path=path_to_config)
    print("Configs loaded")

    data: DataFrame = read_csv(configs["ground_truth_path"])
    # NOTE: utterances have .mp4 at the end. I am removing it
    # NOTE: I could have also used used [: -4] to remove the last 4 characters
    data["utterance"] = data["utterance"].apply(lambda x: x.split(".")[0])
    # NOTE: selecting only video and utterance columns
    data = data.iloc[:, 3:5]

    data["img_list"] = data.apply(create_img_list, axis=1, args=(configs["data_path"],))
    output_path: str = join_paths("support_tables/", configs["output_name"])
    data.to_csv(
        output_path, index=True, sep=" "
    )
    print(f"File saved at {output_path}")


if __name__ == "__main__":
    main()
