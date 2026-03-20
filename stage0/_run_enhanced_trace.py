# -*- coding: utf-8 -*-

from create_daily_enhanced_trace import *
import logging, sys, gc
gc.collect()

from _trace_settings import get_config

cfg = get_config("enhanced")   
if __name__ == "__main__":          # ← ADD THIS GUARD
    all_data = CreateDailyEnhancedTRACE(**cfg)