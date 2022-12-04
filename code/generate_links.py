import pandas as pd
import numpy as np


def generate_true_links(df):
    # although the match_id column is included in the original df to imply the true links,
    # this function will create the true_link object identical to the true_links properties
    # of recordlinkage toolkit, in order to exploit "Compare.compute()" from that toolkit
    # in extract_function() for extracting features quicker.
    # This process should be deprecated in the future release of the UNSW toolkit.
    df["rec_id"] = df.index.values.tolist()
    indices_1 = []
    indices_2 = []
    processed = 0
    for match_id in df["match_id"].unique():
        if match_id != -1:
            processed = processed + 1
            # print("In routine generate_true_links(), count =", processed)
            # clear_output(wait=True)
            linkages = df.loc[df["match_id"] == match_id]
            for j in range(len(linkages) - 1):
                for k in range(j + 1, len(linkages)):
                    indices_1 = indices_1 + [linkages.iloc[j]["rec_id"]]
                    indices_2 = indices_2 + [linkages.iloc[k]["rec_id"]]
    links = pd.MultiIndex.from_arrays([indices_1, indices_2])
    return links


def generate_false_links(df, size):
    # A counterpart of generate_true_links(), with the purpose to generate random false pairs
    # for training. The number of false pairs in specified as "size".
    df["rec_id"] = df.index.values.tolist()
    indices_1 = []
    indices_2 = []
    unique_match_id = df["match_id"].unique()
    for j in range(size):
        false_pair_ids = np.random.choice(unique_match_id, 2)
        candidate_1_cluster = df.loc[df["match_id"] == false_pair_ids[0]]
        candidate_1 = candidate_1_cluster.iloc[
            np.random.choice(range(len(candidate_1_cluster)))
        ]
        candidate_2_cluster = df.loc[df["match_id"] == false_pair_ids[1]]
        candidate_2 = candidate_2_cluster.iloc[
            np.random.choice(range(len(candidate_2_cluster)))
        ]
        indices_1 = indices_1 + [candidate_1["rec_id"]]
        indices_2 = indices_2 + [candidate_2["rec_id"]]
    links = pd.MultiIndex.from_arrays([indices_1, indices_2])
    return links
