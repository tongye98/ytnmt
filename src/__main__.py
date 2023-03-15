import argparse
from src.training import train 
from src.testing import test 
from src.metrics import test_metrics

def main():
    parser = argparse.ArgumentParser("Transformer")

    parser.add_argument("mode", choices=["train","test", "metric"])
    parser.add_argument("config_path", type=str, help="path to a config yaml file")
    parser.add_argument("--ckpt", type=str, help="model checkpoint for prediction")

    args = parser.parse_args()

    if args.mode == "train":
        train(cfg_file=args.config_path)

    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt_path=args.ckpt)
    
    elif args.mode == "metric":
        test_metrics()
        
    else:
        raise ValueError("Unkonwn mode!")

if __name__ == "__main__":
    main()