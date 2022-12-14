import yaml

def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)

    return content

def get_timestamp(name):
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{name}_at_{timestamp}" 

    return unique_name