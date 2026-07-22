import os
import argparse
import yaml
from pathlib import Path

# Get directory where this config.py is located
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

def load_config():
    """Loads configuration from config.yaml and overrides with CLI arguments."""
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # --- Parse Command Line Arguments for quick overrides ---
    parser = argparse.ArgumentParser(description="Face Clustering & Embedding Benchmark", add_help=False)
    # Use argparse.ArgumentParser(add_help=False) to avoid conflict when imported/used elsewhere
    # or just parse known args
    parser.add_argument("--dataset_dir", type=str, help="Path to dataset directory (containing pool/ and ground_truth.csv)")
    parser.add_argument("--model_name", type=str, choices=["buffalo_l", "buffalo_m", "buffalo_sc", "finetuned", "adaface"], help="InsightFace model pack, 'finetuned', or 'adaface'")
    parser.add_argument("--mode", type=str, choices=["raw", "cropped"], help="Extraction mode ('raw' or 'cropped')")
    parser.add_argument("--output_dir", type=str, help="Output directory for embeddings")
    parser.add_argument("--results_dir", type=str, help="Output directory for results/reports")
    parser.add_argument("--limit_persons", type=int, default=None, help="Limit number of persons to process for quick testing")
    
    # parse_known_args prevents crash when running other scripts that have their own args
    args, _ = parser.parse_known_args()
    
    # --- Merge CLI arguments into config dict ---
    if args.dataset_dir:
        config["dataset"]["dir"] = args.dataset_dir
    if args.model_name:
        config["extraction"]["model_name"] = args.model_name
    if args.mode:
        config["extraction"]["mode"] = args.mode
    if args.output_dir:
        config["extraction"]["output_dir"] = args.output_dir
    if args.results_dir:
        config["clustering"]["output_dir"] = args.results_dir
    if args.limit_persons:
        config["extraction"]["limit_persons"] = args.limit_persons
    else:
        config["extraction"]["limit_persons"] = None
        
    # --- Resolve Paths to absolute paths ---
    dataset_path = Path(config["dataset"]["dir"])
    if not dataset_path.is_absolute():
        dataset_path = (PROJECT_ROOT / dataset_path).resolve()
    config["dataset"]["dir_path"] = dataset_path
    
    emb_out_path = Path(config["extraction"]["output_dir"])
    if not emb_out_path.is_absolute():
        emb_out_path = (PROJECT_ROOT / emb_out_path).resolve()
    config["extraction"]["output_path"] = emb_out_path
    
    res_out_path = Path(config["clustering"]["output_dir"])
    if not res_out_path.is_absolute():
        res_out_path = (PROJECT_ROOT / res_out_path).resolve()
    config["clustering"]["output_path"] = res_out_path

    # Helper paths for embeddings, speed metrics, and reports
    model_name = config["extraction"]["model_name"]
    mode = config["extraction"]["mode"]
    
    if model_name == "finetuned":
        config["extraction"]["custom_rec_onnx"] = str((SCRIPT_DIR.parent / "model" / "best_model.onnx").resolve())
    else:
        config["extraction"]["custom_rec_onnx"] = None

    # Handle AdaFace path resolution
    adaface_ckpt = config["extraction"].get("adaface_ckpt", "model/adaface_ir50_ms1mv2.ckpt")
    adaface_ckpt_path = Path(adaface_ckpt)
    if not adaface_ckpt_path.is_absolute():
        adaface_ckpt_path = (PROJECT_ROOT / adaface_ckpt_path).resolve()
    config["extraction"]["adaface_ckpt_path"] = str(adaface_ckpt_path)

    config["extraction"]["embeddings_file"] = emb_out_path / f"{model_name}_{mode}_embeddings.npz"
    config["extraction"]["speed_file"] = emb_out_path / f"{model_name}_{mode}_speed.npz"
    config["clustering"]["report_md"] = res_out_path / f"{model_name}_{mode}_report.md"
    config["clustering"]["report_json"] = res_out_path / f"{model_name}_{mode}_report.json"
    
    return config

# Global config object loaded when imported
cfg = load_config()
