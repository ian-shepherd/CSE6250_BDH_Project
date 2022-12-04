import recordlinkage as rl
import pandas as pd
import numpy as np
from recordlinkage.datasets import load_febrl4, load_febrl1, load_febrl2, load_febrl3
from IPython.display import clear_output


def preprocess(df):
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["day"] = df["date_of_birth"].dt.strftime("%d")
    df["month"] = df["date_of_birth"].dt.strftime("%m")
    df["year"] = df["date_of_birth"].dt.strftime("%Y")

    df["postcode"] = df["postcode"].fillna("0000")
    df["postcode"] = df["postcode"].astype(int)

    df["street_number"] = df["street_number"].fillna("0")
    df["street_number"] = df["street_number"].astype(int)

    df = df.drop(["soc_sec_id", "date_of_birth"], axis=1)

    for col in ["surname", "given_name", "address_1", "address_2", "day", "month"]:
        df[col] = df[col].fillna("")
        df[col] = df[col].astype(str)

    # df["match_id"] = range(len(df))

    all_fields = df.columns.values.tolist()
    # all_fields.remove('rec_id')

    return df, all_fields


def processed(df, true_links_ab):
    df["match_id"] = [-1] * len(df)

    for i in range(len(true_links_ab)):
        k0 = true_links_ab[i][0]
        k1 = true_links_ab[i][1]
        df.loc[k0, "match_id"] = i
        df.loc[k1, "match_id"] = i
        # print("Processed:", i)

    return df


def generate_files(PATH_FILES, type_set):

    if type_set == "train":
        df, true_links_ab = load_febrl3(return_links=True)  # use this for load_febrl4
        OUTPUT_FILE_NAME = "febrl3_rep.csv"
    elif type_set == "test":
        df_a, df_b, true_links_ab = load_febrl4(return_links=True)
        df = df_a.append(df_b)  # use for load_febrl4
        OUTPUT_FILE_NAME = "febrl4_rep.csv"

    print("df:", df)
    print("true_links_ab:", true_links_ab)

    outputpath = PATH_FILES + OUTPUT_FILE_NAME
    # df, true_links_ab = load_febrl3(return_links=True)
    # WARNING: load_febrl2 and load_febrl3 does not suit this code, need further process

    df, all_fields = preprocess(df)
    df = processed(df, true_links_ab)
    df.to_csv(outputpath, index=True)
    return outputpath


if __name__ == "__main__":

    # PATH_FILES = "../data/FEBRL/"
    PATH_FILES = "./data/FEBRL/"
    generate_files(PATH_FILES, "train")
    generate_files(PATH_FILES, "test")
