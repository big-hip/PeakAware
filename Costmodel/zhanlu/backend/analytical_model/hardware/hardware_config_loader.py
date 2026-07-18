from functools import lru_cache
import os
import json
import warnings

warnings.filterwarnings("once", category=DeprecationWarning)
cached_config = {} #  缓存减少访问IO次数


def load_hardware_config(
    chip_model, topo_name, chip_dir="chip_configs", topo_dir="topo_configs"
):
    """
    根据芯片型号和拓扑名称加载硬件配置
    """

    current_file_path = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(current_file_path)

    chip_path = os.path.join(current_dir_path, chip_dir, f"{chip_model}.json")
    topo_path = os.path.join(current_dir_path, topo_dir, f"{topo_name}.json")

    if not os.path.exists(chip_path):
        raise FileNotFoundError(f"芯片配置文件不存在: {chip_path}")
    if not os.path.exists(topo_path):
        raise FileNotFoundError(f"拓扑配置文件不存在: {topo_path}")

    key = chip_path + topo_path
    if key in cached_config.keys():
        return cached_config[key]

    config_dict = {}

    try:
        with open(chip_path, "r") as f:
            chip_config = json.load(f)
            config_dict.update({"chip_config": chip_config})
    except json.JSONDecodeError:
        raise ValueError(f"芯片配置文件格式错误: {chip_path}")

    try:
        with open(topo_path, "r") as f:
            topo_config = json.load(f)
            config_dict.update({"topo_config": topo_config})
    except json.JSONDecodeError:
        raise ValueError(f"拓扑配置文件格式错误: {topo_path}")

    cached_config[key] = config_dict

    return config_dict


@lru_cache(maxsize=None)
def load_vector_config(chip_model, chip_dir="chip_configs"):
    current_file_path = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(current_file_path)

    chip_path = os.path.join(current_dir_path, chip_dir, f"{chip_model}.json")

    if not os.path.exists(chip_path):
        raise FileNotFoundError(f"芯片配置文件不存在: {chip_path}")

    config_dict = {}

    try:
        with open(chip_path, "r") as f:
            chip_config = json.load(f)
            config_dict["vector"] = chip_config["vector"]
            config_dict["vector"]["compute"] = chip_config["compute"]["vector"]
    except KeyError:
        DEFAULT_FILE = "A3"
        warnings.warn("instruction detail use A3 config", DeprecationWarning, stacklevel=2)
        return load_vector_config(DEFAULT_FILE)
    except json.JSONDecodeError:
        raise ValueError(f"芯片配置文件格式错误: {chip_path}")

    return config_dict["vector"]


@lru_cache(maxsize=None)
def load_vector_OPs(chip_model, chip_dir="chip_configs"):
    current_file_path = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(current_file_path)

    chip_path = os.path.join(current_dir_path, chip_dir, f"{chip_model}.json")

    if not os.path.exists(chip_path):
        raise FileNotFoundError(f"芯片配置文件不存在: {chip_path}")

    with open(chip_path, "r") as f:
        chip_config = json.load(f)
        vector_OPs = chip_config["compute"]["vector"]

    return vector_OPs
