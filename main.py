import sys
sys.path.insert(0, "stage0")
sys.path.insert(0, "stage1")

from wrds_trace_download import *
import logging, gc

if __name__ == '__main__':

    # --- TRACE downloads/update ---

    dl = WRDSDownloader(wrds_username = "") # Please login WRDS website to get verification first.
    dl.run_all()    

    # --- Stage 0: TRACE cleaning ---
    from _trace_settings import get_config
    from create_daily_standard_trace import CreateDailyStandardTRACE
    from create_daily_enhanced_trace import CreateDailyEnhancedTRACE

    logging.info("=== Running 144a trace ===")
    all_data_144a = CreateDailyStandardTRACE(**get_config("144a"))
    gc.collect()

    logging.info("=== Running standard trace ===")
    all_data_standard = CreateDailyStandardTRACE(**get_config("standard"))
    gc.collect()

    logging.info("=== Running enhanced trace ===")
    all_data_enhanced = CreateDailyEnhancedTRACE(**get_config("enhanced"))
    gc.collect()    

    # --- Stage 1: TRACE characteristics ---
    from create_daily_stage1 import run_stage1
    from _stage1_settings import get_config, print_config_summary

    logging.info("=== Running stage1 ===")
    config = get_config()
    print_config_summary(config)
    try:
        processor = run_stage1(config)
        print("\nStage 1 processing completed successfully!")
    except Exception as e:
        logging.exception("Stage 1 pipeline error:")
        raise
    gc.collect()