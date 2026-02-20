import os
import json
import argparse

parser = argparse.ArgumentParser(description="convert")
parser.add_argument("--ckpt_path", type=str, help="path of modeling", default=None)
parser.add_argument("--base_arch_path", type=str, help="path of modeling", default=None)
args = parser.parse_args()
if args.base_arch_path is None:
    args.base_arch_path = args.ckpt_path

config_path = os.path.join(args.ckpt_path, 'config.json')

with open(os.path.join(args.base_arch_path, 'config.json'), 'r') as file:
    base_arch_config = json.load(file)

with open(config_path, 'r') as file:
    data = json.load(file)
    model_config = config_path.split('/')[-2].split('-')
    
    if "ConcatProj" in model_config:
        data["moe_concat_proj"] = True
    else:
        data["moe_concat_proj"] = False
    data['R_I'] = 0
    data['R_O'] = 0
    for c in model_config:
        if c.startswith('EI'):
            data['moe_intermediate_size'] = int(c.removeprefix('EI'))
        if c.startswith('SI'):
            shared_hidden_size = int(c.removeprefix('SI'))
            data['shared_expert_intermediate_size'] = shared_hidden_size
        elif c.startswith('NumExpert'):
            data['num_experts'] = int(c.removeprefix('NumExpert'))
        elif c.startswith('TOP'):
            data['num_experts_per_tok'] = int(c.removeprefix('TOP'))
        elif c.startswith('G_I'):
            data['G_I'] = int(c.removeprefix('G_I'))
        elif c.startswith('G_O'):
            data['G_O'] = int(c.removeprefix('G_O'))
        elif c.startswith('R_I'):
            data['R_I'] = int(c.removeprefix('R_I'))
        elif c.startswith('R_O'):
            data['R_O'] = int(c.removeprefix('R_O'))

    for key in ["hidden_size", "intermediate_size", "max_position_embeddings", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rms_norm_eps", "vocab_size", "max_window_layers"]:
        data[key] = base_arch_config[key]

    has_py_file = False
    for filename in os.listdir(args.ckpt_path):
        if filename.endswith(".py"):
            has_py_file = True
            modeling_filename = filename.split('.')[0]
    if has_py_file:
        print(f"Found modeling file: {modeling_filename}")
        import sys, inspect, importlib
        sys.path.append(args.ckpt_path)
        module = importlib.import_module(modeling_filename)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if not obj.__module__ == module.__name__:
                continue
            if name.endswith("CausalLM"):
                print(f"Found CausalLM with name: {name}")
                data['architectures'] = name
            elif name.endswith("Config"):
                print(f"Found Config with name: {name}")
                data['model_type'] = obj.model_type

with open(config_path, 'w') as file:
    json.dump(data, file, indent=4)
