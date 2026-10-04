import argparse
import configparser


def load_config(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", help="Path to the experiment config file")
    parser.add_argument("--config", dest="config_option", help="Path to the experiment config file")
    args = parser.parse_args(argv)

    if args.config_path and args.config_option:
        parser.error("provide the config path once")
    config_path = args.config_path or args.config_option
    if not config_path:
        parser.error("a config file path is required")

    cfg = configparser.ConfigParser()
    if not cfg.read(config_path):
        parser.error("could not read config file: {}".format(config_path))

    return cfg["DEFAULT"], config_path
