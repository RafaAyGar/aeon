"""Functions to load and write datasets."""

__all__ = [
    # Load/download functions
    "load_from_tsfile",
    "load_from_tsf_file",
    "load_from_arff_file",
    "load_from_tsv_file",
    "load_from_timeeval_csv_file",
    "load_anomaly_detection",
    "load_classification",
    "load_forecasting",
    "load_regression",
    "download_all_regression",
    "load_time_series_segmentation_benchmark",
    "load_human_activity_segmentation_datasets",
    # Write functions
    "write_to_tsfile",
    "write_to_tsf_file",
    "write_to_arff_file",
    "write_results_to_uea_format",
    # Single problem loaders
    "load_airline",
    "load_arrow_head",
    "load_gunpoint",
    "load_basic_motions",
    "load_osuleaf",
    "load_italy_power_demand",
    "load_japanese_vowels",
    "load_plaid",
    "load_longley",
    "load_lynx",
    "load_shampoo_sales",
    "load_unit_test",
    "load_uschange",
    "load_PBS_dataset",
    "load_japanese_vowels",
    "load_gun_point_segmentation",
    "load_electric_devices_segmentation",
    "load_acsf1",
    "load_macroeconomic",
    "load_unit_test_tsf",
    "load_solar",
    "load_cardano_sentiment",
    "load_covid_3month",
    "load_kdd_tsad_135",
    "load_daphnet_s06r02e0",
    "load_ecg_diff_count_3",
    "load_atrial_fibrillation",
    # legacy load functions
    "load_from_arff_to_dataframe",
    "load_from_ucr_tsv_to_dataframe",
    "load_from_tsfile_to_dataframe",
    "get_dataset_meta_data",
]

from aeon.datasets._data_loaders import (
    download_all_regression,
    get_dataset_meta_data,
    load_classification,
    load_forecasting,
    load_from_arff_file,
    load_from_tsf_file,
    load_from_tsfile,
    load_from_tsv_file,
    load_regression,
)
from aeon.datasets._data_writers import (
    write_results_to_uea_format,
    write_to_arff_file,
    write_to_tsf_file,
    write_to_tsfile,
)
from aeon.datasets._dataframe_loaders import (
    load_from_arff_to_dataframe,
    load_from_tsfile_to_dataframe,
    load_from_ucr_tsv_to_dataframe,
)
from aeon.datasets._single_problem_loaders import (
    load_acsf1,
    load_airline,
    load_arrow_head,
    load_atrial_fibrillation,
    load_basic_motions,
    load_cardano_sentiment,
    load_covid_3month,
    load_electric_devices_segmentation,
    load_gun_point_segmentation,
    load_gunpoint,
    load_italy_power_demand,
    load_japanese_vowels,
    load_longley,
    load_lynx,
    load_macroeconomic,
    load_osuleaf,
    load_PBS_dataset,
    load_plaid,
    load_shampoo_sales,
    load_solar,
    load_unit_test,
    load_unit_test_tsf,
    load_uschange,
)
from aeon.datasets._tsad_data_loaders import (
    load_anomaly_detection,
    load_daphnet_s06r02e0,
    load_ecg_diff_count_3,
    load_from_timeeval_csv_file,
    load_kdd_tsad_135,
)
from aeon.datasets._tss_data_loaders import (
    load_human_activity_segmentation_datasets,
    load_time_series_segmentation_benchmark,
)
