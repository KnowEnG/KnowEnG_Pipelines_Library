"""
    This module serves as a connecting function between front end and back end.
    It validates/cleans the user spreadsheet data and returns a boolean value to
    indicate if the user spreadsheet is valid or not. 
"""
import pandas

def check_input_value_for_gene_prioritazion(data_frame, phenotype_df):
    """
    This input value check is specifically designed for gene_priorization_pipeline.
    1. user spreadsheet contains real number.
    2. phenotype data contains only positive number, including 0.

    Args:
        data_frame: user spreadsheet as data frame
        phenotype_df: phenotype data as data frame
        run_parameters: configuration as data dictionary

    Returns:
        data_frame_trimed: Either None or trimed data frame will be returned for calling function
        phenotype_trimed: Either None or trimed data frame will be returned for calling function
        message: A message indicates the status of current check
    """
    # drops column which contains NA in data_frame
    data_frame_dropna = data_frame.dropna(axis=1)

    if data_frame_dropna.empty:
        return None, None, "Data frame is empty after remove NA."

    # checks real number negative to positive infinite
    data_frame_check = data_frame_dropna.applymap(lambda x: isinstance(x, (int, float)))

    if False in data_frame_check:
        return None, None, "Found not numeric value in user spreadsheet."

    # drops columns with NA value in phenotype dataframe (sample x phenotype)
    phenotype_df_dropna = phenotype_df.dropna(axis=1)

    phenotype_header = list(phenotype_df_dropna.columns.values)
    data_frame_header = list(data_frame.columns.values)

    # common headers between phenotype data and user spreadsheet
    common_headers = list(set(phenotype_header) & set(data_frame_header))

    if not common_headers:
        return None, None, "Cannot find intersection between user spreadsheet column and phenotype data."

    # sorts common_headers to be in same order
    common_headers.sort()

    # select common column to process, this operation will reorder the column
    data_frame_trimed = data_frame_dropna[common_headers]
    phenotype_trimed = phenotype_df_dropna[common_headers]

    if data_frame_trimed.empty:
        return None, None, "Cannot find valid value in user spreadsheet."

    return data_frame_trimed, phenotype_trimed, "Passed input value validation."

