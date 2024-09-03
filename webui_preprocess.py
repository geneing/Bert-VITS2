import gradio as gr
import webbrowser
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


def resample(data_dir):
    assert data_dir != "", "Dataset name cannot be empty"
    start_path, _, _, _, config_path = get_path(data_dir)
    in_dir = os.path.join(start_path, "raw")
    out_dir = os.path.join(start_path, "wavs")
    subprocess.run(
        f"python resample_legacy.py "
        f"--sr 44100 "
        f"--in_dir {in_dir} "
        f"--out_dir {out_dir} ",
        shell=True,
    )
    return "Audio files preprocessing completed"


def preprocess_text(data_dir):
    assert data_dir != "", "Dataset name cannot be empty"
    start_path, lbl_path, train_path, val_path, config_path = get_path(data_dir)
    lines = open(lbl_path, "r", encoding="utf-8").readlines()
    with open(lbl_path, "w", encoding="utf-8") as f:
        for line in lines:
            path, spk, language, text = line.strip().split("|")
            path = os.path.join(start_path, "wavs", os.path.basename(path)).replace(
                "\\", "/"
            )
            f.writelines(f"{path}|{spk}|{language}|{text}\n")
    subprocess.run(
        f"python preprocess_text.py "
        f"--transcription-path {lbl_path} "
        f"--train-path {train_path} "
        f"--val-path {val_path} "
        f"--config-path {config_path}",
        shell=True,
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
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                _ = gr.Markdown(
                    value="# Bert-VITS2 Data Preprocessing\n"
                    "## Preparation:\n"
                    "Download BERT and WavLM models:\n"
                    "- Chinese RoBERTa\n"
                    "- Japanese DeBERTa\n"
                    "- English DeBERTa\n"
                    "- WavLM\n"
                    "\n"
                    "Place the BERT models in the `bert` folder and the WavLM model in the `slm` folder, overwriting the folders with the same name.\n"
                    "\n"
                    "Data preparation:\n"
                    "Place the data in the data folder, organized as follows:\n"
                    "\n"
                    "```\n"
                    "├── data\n"
                    "│   ├── {your dataset name}\n"
                    "│   │   ├── esd.list\n"
                    "│   │   ├── raw\n"
                    "│   │   │   ├── ****.wav\n"
                    "│   │   │   ├── ****.wav\n"
                    "│   │   │   ├── ...\n"
                    "```\n"
                    "\n"
                    "In the `raw` folder, save all audio files. The `esd.list` file is the label text file, formatted as\n"
                    "\n"
                    "```\n"
                    "****.wav|{speaker name}|{language ID}|{label text}\n"
                    "```\n"
                    "\n"
                    "For example:\n"
                    "```\n"
                    "vo_ABDLQ001_1_paimon_02.wav|Paimon|ZH|Nothing, nothing, it's just that he's usually standing here, it's a bit strange.\n"
                    "noa_501_0001.wav|NOA|JP|Yes, I think it's very important not to let your guard down.\n"
                    "Albedo_vo_ABDLQ002_4_albedo_01.wav|Albedo|EN|Who are you? Why did you alarm them?\n"
                    "...\n"
                    "```\n"
                )
                data_dir = gr.Textbox(
                    label="Dataset Name",
                    placeholder="The name of the folder where your dataset is placed under the data folder, e.g., if data/genshin, then fill in genshin",
                )
                info = gr.Textbox(label="Status Information")
                _ = gr.Markdown(value="## Step 1: Generate Configuration File")
                with gr.Row():
                    batch_size = gr.Slider(
                        label="Batch Size: 24 GB VRAM can use 12",
                        value=8,
                        minimum=1,
                        maximum=64,
                        step=1,
                    )
                    generate_config_btn = gr.Button(value="Execute", variant="primary")
                _ = gr.Markdown(value="## Step 2: Preprocess Audio Files")
                resample_btn = gr.Button(value="Execute", variant="primary")
                _ = gr.Markdown(value="## Step 3: Preprocess Label Files")
                preprocess_text_btn = gr.Button(value="Execute", variant="primary")
                _ = gr.Markdown(value="## Step 4: Generate BERT Feature Files")
                bert_gen_btn = gr.Button(value="Execute", variant="primary")
                _ = gr.Markdown(
                    value="## Train and Deploy Model:\n"
                    "Modify the `dataset_path` item in the `config.yml` file in the root directory to `data/{your dataset name}`\n"
                    "- Training: Place the pre-trained model files (`D_0.pth`, `DUR_0.pth`, `WD_0.pth`, and `G_0.pth`) in the `data/{your dataset name}/models` folder, and execute the `torchrun --nproc_per_node=1 train_ms.py` command (for multi-card operation, refer to the commands in `run_MnodesAndMgpus.sh`).\n"
                    "- Deployment: Modify the `model` item under `webui` in the `config.yml` file in the root directory to `models/{weight file name}.pth` (e.g., G_10000.pth), then execute `python webui.py`"
                )

        generate_config_btn.click(
            generate_config, inputs=[data_dir, batch_size], outputs=[info]
        )
        resample_btn.click(resample, inputs=[data_dir])

        preprocess_text_btn.click(preprocess_text, inputs=[data_dir], outputs=[info])
        bert_gen_btn.click(bert_gen, inputs=[data_dir], outputs=[info])

    webbrowser.open("http://127.0.0.1:7860")
    app.launch(share=False, server_port=7860)
