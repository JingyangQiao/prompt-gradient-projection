import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import json
import argparse

from run import run


def load_json(json_path):
    with open(json_path) as json_file:
        params = json.load(json_file)
    return params


def setup_parser():
    parser = argparse.ArgumentParser(description="Task Incremental Learning by CLIP")
    parser.add_argument('--config', type=str, default="", help="Path of json file.")
    return parser


def main():
    args = setup_parser().parse_args()
    params = load_json(args.config)
    args = vars(args)
    args.update(params)
    run(args)


if __name__ == "__main__":
    main()
