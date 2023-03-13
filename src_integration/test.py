import logging 
import torch 
from typing import Dict, List
import time 
from pathlib import Path 
from train import load_config, make_logger
from data import load_data, make_data_loader
from model import build_model
from validate import search, eval_accuracies
from tqdm import tqdm 
logger = logging.getLogger(__name__)

def resolve_ckpt_path(ckpt_path:str, load_model:str, model_dir:Path) -> Path:
    """
    Resolve checkpoint path
    First choose ckpt_path, then choos load_model, 
    then choose model_dir/best.ckpt, final choose model_dir/latest.ckpt
    """
    if ckpt_path is None:
        logger.warning("ckpt_path is not specified.")
        if load_model is None:
            if (model_dir / "best.ckpt").is_file():
                ckpt_path = model_dir / "best.ckpt"
                logger.warning("use best ckpt in model dir!")
            else:
                logger.warning("No ckpt_path, no load_model, no best_model, Please Check!")
                ckpt_path = model_dir / "latest.ckpt"
                logger.warning("use latest ckpt in model dir!")
        else:
            logger.warning("use load_model item in config yaml.")
            ckpt_path = Path(load_model)
    return Path(ckpt_path)

def load_model_checkpoint(path:Path, device:torch.device) -> Dict:
    """
    Load model from saved model checkpoint
    """
    assert path.is_file(), f"model checkpoint {path} not found!"
    model_checkpoint = torch.load(path.as_posix(), map_location=device)
    logger.info("Load model from %s.", path.resolve())
    return model_checkpoint

def write_model_generated_to_file(path:Path, array: List[str]) -> None:
    """
    Write list of strings to file.
    array: list of strings.
    """
    with path.open("w", encoding="utf-8") as fg:
        for entry in array:
            fg.write(f"{entry}\n")


def test(cfg_file: str) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating translations.
    """
    cfg = load_config(Path(cfg_file))

    model_dir = cfg["training"].get("model_dir", None)
    assert model_dir is not None

    make_logger(model_dir, mode="test")

    load_model = cfg["training"].get("load_model", None)
    use_cuda = cfg["training"].get("use_cuda", False) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    seed = cfg["training"].get("random_seed", 980820)
    batch_size = cfg["testing"].get("batch_size", 32)
    num_workers = cfg["training"].get("num_workers", 4)
    beam_size = cfg["testing"].get("beam_size", 4)

    # load data
    train_dataset, valid_dataset, test_dataset, vocab_info = load_data(data_cfg=cfg["data"])

    # build model
    model = build_model(model_cfg=cfg["model"], vocab_info=vocab_info)

    # when checkpoint is not specified, take latest(best) from model_dir
    ckpt_path = resolve_ckpt_path(ckpt_path, load_model, Path(model_dir))
    logger.info("ckpt_path = {}".format(ckpt_path))

    # load model checkpoint 
    model_checkpoint = load_model_checkpoint(path=ckpt_path, device=device)

    # restore model and optimizer parameters
    model.load_state_dict(model_checkpoint["model_state"])

    # model to GPU
    if device.type == "cuda":
        model.to(device)
    
    # Test 
    dataset_to_test = {"valid": valid_dataset, "test":test_dataset}
    for dataset_name, dataset in dataset_to_test.item():
        if dataset_name == "valid":
            continue
        if dataset is not None: 
            logger.info("Starting testing on %s dataset...", dataset_name)
            test_start_time = time.time()
            test_loader = make_data_loader(dataset=dataset, sampler_seed=seed, shuffle=False,
                            batch_size=batch_size, num_workers=num_workers, mode="test")
            
            model.eval()

            all_test_outputs = []
            all_test_probability = []
            all_test_attention = []
            eval_scores = {}

            for batch_data in tqdm(test_loader, desc="Testing"):
                batch_data.to(device)
                stacked_output, stacked_probability, stacked_attention = search(batch_data, model, cfg)

                all_test_outputs.extend(stacked_output)
                all_test_probability.extend(stacked_probability)
                all_test_attention.extend(stacked_attention)
            
            text_vocab = vocab_info["trg_vocab"]["self"]
            model_generated = text_vocab.arrays_to_sentence(arrays=all_test_outputs, cut_at_eos=True, skip_pad=True)
            model_generated = [" ".join(output) for output in model_generated]

            target_truth = dataset.target_truth

            test_duration_time = time.time() - test_start_time

            bleu, rouge_l, meteor = eval_accuracies(model_generated, target_truth)
            eval_scores["bleu"] = bleu
            eval_scores["rouge_l"] = rouge_l
            eval_scores["meteor"] = meteor

            metrics_string = "Bleu={}, rouge_l={}, meteor={}".format(bleu, rouge_l, meteor)
            logger.info("Evaluation result({}) {}, evaluation time = {:.2f}[sec]".format(
                "Beam Search" if beam_size > 1 else "Greedy Search", metrics_string))
            
            output_file_path = Path(model_dir) / "{}.test_out".format(dataset_name)
            write_model_generated_to_file(output_file_path, model_generated)


if __name__ == "__main__":
    cfg_file = "test.yaml"
    test(cfg_file)