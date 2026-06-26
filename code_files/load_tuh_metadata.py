import os
import glob
import argparse
import pandas as pd
# pip install sqlalchemy psycopg2-binary   # install if needed

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Float,
    Integer,
    select,
    func
)
from sqlalchemy.exc import SQLAlchemyError


# database connection details
DB_HOST = "localhost"
DB_NAME = "tuheeg"
DB_USER = "tuheeg_user"
DB_PASSWORD = "tuheeg"
DB_PORT = 5432

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# create engine
engine = create_engine(DATABASE_URL, echo=False, future=True)
metadata = MetaData(schema="public")


# defining the 3 tables here so that script can create them if needed
patient_table = Table(
    "patient",
    metadata,
    Column("dir_type", String),
    Column("patient_id", String),
    Column("session_id", String),
    Column("tcp_ref", String),
    Column("t_id", String),
    Column("session_date", String),
    Column("file_path", String),
    Column("file_type", String),
)

channel_annotations_table = Table(
    "channel_annotations",
    metadata,
    Column("channel", String),
    Column("start_time", Float),
    Column("stop_time", Float),
    Column("label", String),
    Column("confidence", Float),
    Column("patient_id", String),
    Column("session_id", String),
    Column("t_id", String),
    Column("tcp_ref", String),
    Column("file_path", String),
    Column("ses_date", String),
)

term_annotations_table = Table(
    "term_annotations",
    metadata,
    Column("channel", String),
    Column("start_time", Float),
    Column("stop_time", Float),
    Column("label", String),
    Column("confidence", Float),
    Column("patient_id", String),
    Column("session_id", String),
    Column("t_id", String),
    Column("tcp_ref", String),
    Column("file_path", String),
    Column("ses_date", String),
)


def create_required_items():
    # creating tables if they are not already there
    metadata.create_all(engine)


# function for inserting in patient table
def insert_in_patient(dir_type, patient_id, session_id, tcp_ref, t_id, session_date, file_path, file_type):
    num_rows_before_ins = 0
    num_rows_after_ins = 0
    try:
        with engine.begin() as conn:
            # checking count of rows before insertion
            try:
                qry_count = select(func.count()).select_from(patient_table)
                num_rows_before_ins = conn.execute(qry_count).scalar_one()
            except Exception as ex:
                num_rows_before_ins = 0
                print(ex)

            ins_stmt = patient_table.insert().values(
                dir_type=dir_type,
                patient_id=patient_id,
                session_id=session_id,
                tcp_ref=tcp_ref,
                t_id=t_id,
                session_date=session_date,
                file_path=file_path,
                file_type=file_type,
            )
            conn.execute(ins_stmt)

            try:
                qry_con_str = select(func.count()).select_from(patient_table)
                num_rows_after_ins = conn.execute(qry_con_str).scalar_one()
            except Exception as ex:
                num_rows_after_ins = 0
                print(ex)

        return num_rows_after_ins - num_rows_before_ins

    except SQLAlchemyError as ex:
        print(ex)
        return 0


# function for inserting in channel annotations table
def insert_in_channel_annotations(channel, start_time, stop_time, label, confidence, patient_id, session_id, t_id, tcp_ref, file_path, ses_date):
    num_rows_before_ins = 0
    num_rows_after_ins = 0
    try:
        with engine.begin() as conn:
            # checking count of rows before insertion
            try:
                qry_count = select(func.count()).select_from(channel_annotations_table)
                num_rows_before_ins = conn.execute(qry_count).scalar_one()
            except Exception as ex:
                num_rows_before_ins = 0
                print(ex)

            ins_stmt = channel_annotations_table.insert().values(
                channel=str(channel) if pd.notna(channel) else None,
                start_time=float(start_time) if pd.notna(start_time) else None,
                stop_time=float(stop_time) if pd.notna(stop_time) else None,
                label=str(label) if pd.notna(label) else None,
                confidence=float(confidence) if pd.notna(confidence) else None,
                patient_id=patient_id,
                session_id=session_id,
                t_id=t_id,
                tcp_ref=tcp_ref,
                file_path=file_path,
                ses_date=ses_date,
            )
            conn.execute(ins_stmt)

            try:
                qry_con_str = select(func.count()).select_from(channel_annotations_table)
                num_rows_after_ins = conn.execute(qry_con_str).scalar_one()
            except Exception as ex:
                num_rows_after_ins = 0
                print(ex)

        return num_rows_after_ins - num_rows_before_ins

    except SQLAlchemyError as ex:
        print(ex)
        return 0


# function for inserting in term annotation table
def insert_in_term_annotations(channel, start_time, stop_time, label, confidence, patient_id, session_id, t_id, tcp_ref, file_path, ses_date):
    num_rows_before_ins = 0
    num_rows_after_ins = 0
    try:
        with engine.begin() as conn:
            # checking count of rows before insertion
            try:
                qry_count = select(func.count()).select_from(term_annotations_table)
                num_rows_before_ins = conn.execute(qry_count).scalar_one()
            except Exception as ex:
                num_rows_before_ins = 0
                print(ex)

            ins_stmt = term_annotations_table.insert().values(
                channel=str(channel) if pd.notna(channel) else None,
                start_time=float(start_time) if pd.notna(start_time) else None,
                stop_time=float(stop_time) if pd.notna(stop_time) else None,
                label=str(label) if pd.notna(label) else None,
                confidence=float(confidence) if pd.notna(confidence) else None,
                patient_id=patient_id,
                session_id=session_id,
                t_id=t_id,
                tcp_ref=tcp_ref,
                file_path=file_path,
                ses_date=ses_date,
            )
            conn.execute(ins_stmt)

            try:
                qry_con_str = select(func.count()).select_from(term_annotations_table)
                num_rows_after_ins = conn.execute(qry_con_str).scalar_one()
            except Exception as ex:
                num_rows_after_ins = 0
                print(ex)

        return num_rows_after_ins - num_rows_before_ins

    except SQLAlchemyError as ex:
        print(ex)
        return 0


def process_tuh_eeg_data(input_eeg_filepath):
    all_ctr = 0
    num_rows_inserted = 0
    num_rows_inserted_chnl_ann = 0
    num_rows_inserted_term_ann = 0

    try:
        # input_eeg_filepath should look like:
        # /media/data/TUHEEG/tuh_eeg_seizure/v2.0.0/edf/*
        for dir_path in glob.glob(input_eeg_filepath):
            dir_type = os.path.basename(dir_path)  # train / dev / eval

            patient_glob = os.path.join(dir_path, "*")
            for patient_path in glob.glob(patient_glob):
                patient_id = os.path.basename(patient_path)

                session_glob = os.path.join(patient_path, "*")
                for session_path in glob.glob(session_glob):
                    full_session_id = os.path.basename(session_path)
                    arr = full_session_id.split("_")

                    # expected format: s005_2015_03_11
                    if len(arr) < 4:
                        print(f"session folder name format unexpected: {full_session_id}")
                        continue

                    session_id = arr[0]
                    ses_date = f"{arr[1]}-{arr[2]}-{arr[3]}"

                    tcp_glob = os.path.join(session_path, "*")
                    for tcp_path in glob.glob(tcp_glob):
                        tcp_ref = os.path.basename(tcp_path)

                        file_glob = os.path.join(tcp_path, "*")
                        for end_file in glob.glob(file_glob):
                            if not os.path.isfile(end_file):
                                continue

                            base_name = os.path.basename(end_file)
                            split_name = base_name.split(".")

                            # extracting t_id from file name like aaaaaasgd_s005_t007.edf
                            # or aaaaaasgd_s005_t007.csv / csv_bi
                            name_wo_ext = split_name[0]
                            parts = name_wo_ext.split("_")
                            t_id = parts[-1] if len(parts) > 0 else None

                            ext = None
                            if base_name.endswith(".csv_bi"):
                                ext = "csv_bi"
                            else:
                                ext = os.path.splitext(base_name)[1].replace(".", "")

                            if ext == "csv":
                                file_type = "channel_annotation"

                                # inserting in patient table for this file
                                num_rows_inserted += insert_in_patient(
                                    dir_type, patient_id, session_id, tcp_ref, t_id, ses_date, end_file, file_type
                                )

                                channel_df = pd.read_csv(end_file, skiprows=5)
                                for row in channel_df.itertuples(index=False):
                                    num_rows_inserted_chnl_ann += insert_in_channel_annotations(
                                        getattr(row, "channel", None),
                                        getattr(row, "start_time", None),
                                        getattr(row, "stop_time", None),
                                        getattr(row, "label", None),
                                        getattr(row, "confidence", None),
                                        patient_id,
                                        session_id,
                                        t_id,
                                        tcp_ref,
                                        end_file,
                                        ses_date,
                                    )

                            elif ext == "csv_bi":
                                file_type = "term_annotation"

                                # inserting in patient table for this file
                                num_rows_inserted += insert_in_patient(
                                    dir_type, patient_id, session_id, tcp_ref, t_id, ses_date, end_file, file_type
                                )

                                term_df = pd.read_csv(end_file, skiprows=5)
                                for row in term_df.itertuples(index=False):
                                    num_rows_inserted_term_ann += insert_in_term_annotations(
                                        getattr(row, "channel", None),
                                        getattr(row, "start_time", None),
                                        getattr(row, "stop_time", None),
                                        getattr(row, "label", None),
                                        getattr(row, "confidence", None),
                                        patient_id,
                                        session_id,
                                        t_id,
                                        tcp_ref,
                                        end_file,
                                        ses_date,
                                    )

                            elif ext == "edf":
                                file_type = "edf"

                                # inserting in patient table for edf file also
                                num_rows_inserted += insert_in_patient(
                                    dir_type, patient_id, session_id, tcp_ref, t_id, ses_date, end_file, file_type
                                )

                            all_ctr += 1

    except Exception as ex:
        print(ex)

    print(num_rows_inserted, "rows inserted in patient table")
    print(num_rows_inserted_chnl_ann, "rows inserted in channel_annotations table")
    print(num_rows_inserted_term_ann, "rows inserted in term_annotations table")
    print(all_ctr, "files traversed")


def main():
    parser = argparse.ArgumentParser(description="Load TUH EEG seizure corpus metadata into PostgreSQL using SQLAlchemy.")
    parser.add_argument(
        "input_eeg_filepath",
        type=str,
        help="Input EEG filepath pattern, e.g. '/media/data/TUHEEG/tuh_eeg_seizure/v2.0.0/edf/*'"
    )
    args = parser.parse_args()

    create_required_items()
    process_tuh_eeg_data(args.input_eeg_filepath)


if __name__ == "__main__":
    main()
