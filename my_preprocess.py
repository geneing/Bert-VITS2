import os
import json
import subprocess
import shutil


def get_path(data_dir):
    start_path = os.path.join("./data", data_dir)
    lbl_path = os.path.join(start_path, "esd.list")
    train_path = os.path.join(start_path, "train.list")
    val_path = os.path.join(start_path, "val.list")
    config_path = os.path.join(start_path, "configs", "config.json")
    return start_path, lbl_path, train_path, val_path, config_path


def generate_config(data_dir, batch_size):
    assert data_dir != "", "Dataset name cannot be empty"
    start_path, _, train_path, val_path, config_path = get_path(data_dir)
    if os.path.isfile(config_path):
        config = json.load(open(config_path, "r", encoding="utf-8"))
    else:
        config = json.load(open("configs/config.json", "r", encoding="utf-8"))
    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["train"]["batch_size"] = batch_size
    out_path = os.path.join(start_path, "configs")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    model_path = os.path.join(start_path, "models")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    if not os.path.exists("config.yml"):
        shutil.copy(src="default_config.yml", dst="config.yml")
    return "Configuration file generated successfully"


def resample(in_dir, out_dir):
    assert in_dir != "", "Dataset name cannot be empty"
    # start_path, _, _, _, config_path = get_path(data_dir)
    # in_dir = os.path.join(start_path, "raw")
    # out_dir = os.path.join(start_path, "wavs")
    subprocess.run(
        f"python resample_legacy.py "
        f"--sr 44100 "
        f"--processes 40 "
        f"--in_dir {in_dir} "
        f"--out_dir {out_dir} ",
        shell=True,
        stdout=open("resample.log", "w"),
    )
    return "Audio files preprocessing completed"


def preprocess_text(data_dir):
    assert data_dir != "", "Dataset name cannot be empty"
    start_path, lbl_path, train_path, val_path, config_path = get_path(data_dir)
    lines = open(lbl_path, "r", encoding="utf-8").readlines()
    with open(lbl_path, "w", encoding="utf-8") as f:
        for line in lines:
            path, spk, text = line.strip().split("|")
            language = "EN"
            path = path.replace("/export/eingerman/audio/LibriTTSR/train-clean-460/", "/rhome/eingerman/Projects/DeepLearning/TTS/Bert-VITS2/data/LibriTTSR/wavs/")
            f.writelines(f"{path}|{spk}|{language}|{text}\n")
    subprocess.run(
        f"python preprocess_text.py "
        f"--transcription-path {lbl_path} "
        f"--train-path {train_path} "
        f"--val-path {val_path} "
        f"--config-path {config_path}",
        shell=True,
        stdout=open("preprocess_text.log", "w"),
    )
    return "Label files preprocessing completed"


def bert_gen(data_dir):
    assert data_dir != "", "Dataset name cannot be empty"
    _, _, _, _, config_path = get_path(data_dir)
    subprocess.run(
        f"python bert_gen.py " f"--config {config_path}",
        shell=True,
    )
    return "BERT feature file generation completed"


if __name__ == "__main__":
  
    data_dir = "LibriTTSR"
    batch_size = 12

    ## Train and Deploy Model:\n"
    #Modify the `dataset_path` item in the `config.yml` file in the root directory to `data/{your dataset name}`\n"
    #- Training: Place the pre-trained model files (`D_0.pth`, `DUR_0.pth`, `WD_0.pth`, and `G_0.pth`) in the `data/{your dataset name}/models` folder, and execute the `torchrun --nproc_per_node=1 train_ms.py` command (for multi-card operation, refer to the commands in `run_MnodesAndMgpus.sh`).\n"
    #- Deployment: Modify the `model` item under `webui` in the `config.yml` file in the root directory to `models/{weight file name}.pth` (e.g., G_10000.pth), then execute `python webui.py`"
    # generate_config(data_dir, batch_size)
    # resample("/export/eingerman/audio/LibriTTSR//train-clean-460/", "/rhome/eingerman/Projects/DeepLearning/TTS/Bert-VITS2/data/LibriTTSR/wavs/")
    # preprocess_text(data_dir)
    bert_gen(data_dir)

