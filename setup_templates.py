import sys
import os
import numpy as np
import json
import argparse

from LLMmap.dataset import load_datasets
from LLMmap.inference import load_LLMmap, write_templates
from LLMmap.utility import write_conf_file
from LLMmap import CONF_NAME
from LLMmap.templates import template_generation
from LLMmap import TEMPLATE_NAME


def main():
    parser = argparse.ArgumentParser(description="Generate and export LLMs templates for open LLMmap inference model based on training set.")
    parser.add_argument("model_home_dir", type=str, help="Path to the model home directory")
    parser.add_argument("-y", "--yes", action="store_true", help="Overwrite existing templates without asking")
    parser.add_argument("--threshold_percentile", type=float, default=95.0,
                        help="Percentile of genuine test distances used as the open-set 'unknown model' threshold")
    args = parser.parse_args()

    model_home_dir = args.model_home_dir
    conf, inf = load_LLMmap(model_home_dir, device='cpu')

    if not conf['is_open']:
        print("Applicable to only open-set inference model. Aborting...")
        sys.exit(1)

    conf_for_data = dict(conf)
    conf_for_data.pop('templates', None)
    conf_for_data['inference_model'] = dict(conf['inference_model'])
    (loader_train, loader_test), cache, (dataset_train, dataset_test) = load_datasets(
        conf_for_data,
        mode='classify',
        ks=conf.get('num_istances_dataset', None)
    )
    # label ids must line up with the ones used at training time
    assert cache.llms_map == conf['llms_map'], "Dataset LLM set differs from the one the model was trained on"

    results = template_generation(inf.model, loader_train, loader_test,
                                  l2_normalize=conf.get('l2_normalize_features', False),
                                  threshold_percentile=args.threshold_percentile)
    print(f"Top-1 accuracy on test set: {results['accuracy']:.4f}")
    print(f"Top-3 accuracy on test set: {results['top3_accuracy']:.4f}")
    print(f"Open-set threshold (p{args.threshold_percentile:g} of genuine distances): {results['open_set_threshold']}")
    id2llm = {v: k for k, v in conf['llms_map'].items()}
    worst = sorted(results['per_class_accuracy'].items(), key=lambda kv: kv[1])[:10]
    print("Hardest LLMs:", ", ".join(f"{id2llm[c]}={a:.2f}" for c, a in worst))

    templates_map = {}
    templates = results['templates']
    for i in range(len(templates)):
        llm = id2llm[i]
        templates_map[llm] = templates[i]

    template_out = os.path.join(model_home_dir, TEMPLATE_NAME)

    if os.path.exists(template_out) and not args.yes:
        confirm = input(f"'{template_out}' already exists. Overwrite? (y/n): ")
        if confirm.lower() != 'y':
            print("Aborting.")
            sys.exit(1)

    templates_map = {k: v for k, v in templates_map.items() if np.isfinite(v).all()}
    write_templates(template_out, templates_map)

    # persist the open-set threshold next to the model
    conf_path = os.path.join(model_home_dir, CONF_NAME)
    from LLMmap.utility import read_conf_file
    stored = read_conf_file(conf_path)
    stored['open_set_threshold'] = results['open_set_threshold']
    stored['test_top1_accuracy'] = results['accuracy']
    stored['test_top3_accuracy'] = results['top3_accuracy']
    write_conf_file(conf_path, stored)

    print(f"Templates saved to '{template_out}'")


if __name__ == "__main__":
    main()