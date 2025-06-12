# driver code for single log inference of GRPO
import argparse
from grpo.unsloth_eval_chat import load_model, inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Inference Pipeline")
    parser.add_argument("--model", type=str, help="Path to GRPO model")
    parser.add_argument(
        "--log",
        type=str,
        help="Path to the dataset directory",
        default="data/fyp_data",
    )
    args = parser.parse_args()
    model, tokenizer = load_model(args.model, max_seq_length=2048)
    inference(args.log, model, tokenizer, max_new_tokens=512)
    
