"""
@Desc: 全局配置文件读取
"""

import argparse
import yaml
from typing import Dict, List
import os
import shutil
import sys


class ResampleConfig:
    """Resampling Configuration"""

    def __init__(self, in_dir: str, out_dir: str, sampling_rate: int = 44100):
        self.sampling_rate: int = sampling_rate  # Target sampling rate
        self.in_dir: str = in_dir  # Directory path of audio files to process
        self.out_dir: str = out_dir  # Output path for resampled audio

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        """Create an instance from a dictionary"""

        # Path validity is not checked here, this logic is handled in resample.py
        data["in_dir"] = os.path.join(dataset_path, data["in_dir"])
        data["out_dir"] = os.path.join(dataset_path, data["out_dir"])

        return cls(**data)

class PreprocessTextConfig:
    """Text Preprocessing Configuration"""

    def __init__(
        self,
        transcription_path: str,
        cleaned_path: str,
        train_path: str,
        val_path: str,
        config_path: str,
        val_per_lang: int = 5,
        max_val_total: int = 10000,
        clean: bool = True,
    ):
        self.transcription_path: str = (
            transcription_path  # Path to the original text file, the format should be {wav_path}|{speaker_name}|{language}|{text}.
        )
        self.cleaned_path: str = (
            cleaned_path  # Path for the cleaned text, can be left empty. If empty, it will be generated in the original text directory.
        )
        self.train_path: str = (
            train_path  # Training set path, can be left empty. If empty, it will be generated in the original text directory.
        )
        self.val_path: str = (
            val_path  # Validation set path, can be left empty. If empty, it will be generated in the original text directory.
        )
        self.config_path: str = config_path  # Configuration file path
        self.val_per_lang: int = val_per_lang  # Number of validation samples per speaker
        self.max_val_total: int = (
            max_val_total  # Maximum number of validation samples, excess will be truncated and added to the training set.
        )
        self.clean: bool = clean  # Whether to clean the data

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        """Create an instance from a dictionary"""

        data["transcription_path"] = os.path.join(
            dataset_path, data["transcription_path"]
        )
        if data["cleaned_path"] == "" or data["cleaned_path"] is None:
            data["cleaned_path"] = None
        else:
            data["cleaned_path"] = os.path.join(dataset_path, data["cleaned_path"])
        data["train_path"] = os.path.join(dataset_path, data["train_path"])
        data["val_path"] = os.path.join(dataset_path, data["val_path"])
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        return cls(**data)


class BertGenConfig:
    """Bert Generation Configuration"""

    def __init__(
        self,
        config_path: str,
        num_processes: int = 2,
        device: str = "cuda",
        use_multi_device: bool = False,
    ):
        self.config_path = config_path
        self.num_processes = num_processes
        self.device = device
        self.use_multi_device = use_multi_device

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        return cls(**data)
class EmoGenConfig:
    """EmoGen Configuration"""

    def __init__(
        self,
        config_path: str,
        num_processes: int = 2,
        device: str = "cuda",
        use_multi_device: bool = False,
    ):
        self.config_path = config_path  # Path to the configuration file
        self.num_processes = num_processes  # Number of processes to use
        self.device = device  # Device to use (e.g., "cuda" for GPU)
        self.use_multi_device = use_multi_device  # Whether to use multiple devices

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        """Create an instance from a dictionary"""
        data["config_path"] = os.path.join(dataset_path, data["config_path"])
        return cls(**data)


class TrainMSConfig:
    """Training Configuration"""

    def __init__(
        self,
        config_path: str,
        env: Dict[str, any],
        base: Dict[str, any],
        model: str,
        num_workers: int,
        spec_cache: bool,
        keep_ckpts: int,
    ):
        self.env = env  # Environment variables to load
        self.base = base  # Base model configuration
        self.model = model  # Model storage directory, relative to dataset_path, not the project root
        self.config_path = config_path  # Path to the configuration file
        self.num_workers = num_workers  # Number of workers
        self.spec_cache = spec_cache  # Whether to enable spec cache
        self.keep_ckpts = keep_ckpts  # Number of checkpoints to keep

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        """Create an instance from a dictionary"""
        data["config_path"] = os.path.join(dataset_path, data["config_path"])
        return cls(**data)


class WebUIConfig:
    """WebUI Configuration"""

    def __init__(
        self,
        device: str,
        model: str,
        config_path: str,
        language_identification_library: str,
        port: int = 7860,
        share: bool = False,
        debug: bool = False,
    ):
        self.device = device  # Device to use
        self.model = model  # Model path
        self.config_path = config_path  # Path to the configuration file
        self.port = port  # Port number
        self.share = share  # Whether to share publicly, open to the internet
        self.debug = debug  # Whether to enable debug mode
        self.language_identification_library = language_identification_library  # Language identification library

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        """Create an instance from a dictionary"""
        data["config_path"] = os.path.join(dataset_path, data["config_path"])
        data["model"] = os.path.join(dataset_path, data["model"])
        return cls(**data)
    
class ServerConfig:
    def __init__(
        self, models: List[Dict[str, any]], port: int = 5000, device: str = "cuda"
    ):
        self.models = models  # Configuration for all models to be loaded
        self.port = port  # Port number
        self.device = device  # Default device for models

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


class TranslateConfig:
    """Translation API Configuration"""

    def __init__(self, app_key: str, secret_key: str):
        self.app_key = app_key
        self.secret_key = secret_key

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


class Config:
    def __init__(self, config_path: str):
        if not os.path.isfile(config_path) and os.path.isfile("default_config.yml"):
            shutil.copy(src="default_config.yml", dst=config_path)
            print(
                f"Configuration file {config_path} has been generated based on the default configuration file default_config.yml. Please configure according to the instructions in the configuration file before rerunning."
            )
            print("Unless necessary, please do not modify default_config.yml or back it up.")
            sys.exit(0)
        with open(file=config_path, mode="r", encoding="utf-8") as file:
            yaml_config = yaml.safe_load(file.read())
            dataset_path = yaml_config["dataset_path"]
            openi_token = yaml_config["openi_token"]
            self.dataset_path = dataset_path
            self.mirror = yaml_config["mirror"]
            self.openi_token = openi_token
            self.resample_config = ResampleConfig.from_dict(
                dataset_path, yaml_config["resample"]
            )
            self.preprocess_text_config = PreprocessTextConfig.from_dict(
                dataset_path, yaml_config["preprocess_text"]
            )
            self.bert_gen_config = BertGenConfig.from_dict(
                dataset_path, yaml_config["bert_gen"]
            )
            self.emo_gen_config = EmoGenConfig.from_dict(
                dataset_path, yaml_config["emo_gen"]
            )
            self.train_ms_config = TrainMSConfig.from_dict(
                dataset_path, yaml_config["train_ms"]
            )
            self.webui_config = WebUIConfig.from_dict(
                dataset_path, yaml_config["webui"]
            )
            self.server_config = ServerConfig.from_dict(
                yaml_config["server"]
            )
            self.translate_config = TranslateConfig.from_dict(
                yaml_config["translate"]
            )


parser = argparse.ArgumentParser()
# To avoid conflict with the previous config.json, it has been renamed as follows
parser.add_argument("-y", "--yml_config", type=str, default="config.yml")
args, _ = parser.parse_known_args()
config = Config(args.yml_config)