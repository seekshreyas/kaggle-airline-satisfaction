#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logging configured from configs/log_config.yaml (logger "alfred").
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

import logging
import logging.config

import yaml

from src.utils.paths import project_root


def get_logger(name: str = "alfred") -> logging.Logger:
    """Configure logging once from configs/log_config.yaml (creates logs/) and return a logger."""
    root = project_root()
    if not getattr(get_logger, "_configured", False):
        cfg = yaml.safe_load((root / "configs" / "log_config.yaml").read_text())
        for handler in cfg.get("handlers", {}).values():
            if "filename" in handler:
                (root / handler["filename"]).parent.mkdir(parents=True, exist_ok=True)
                handler["filename"] = str(root / handler["filename"])
        logging.config.dictConfig(cfg)
        get_logger._configured = True
    return logging.getLogger(name)
