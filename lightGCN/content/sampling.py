import pandas as pd
import numpy as np
from numba import njit
import random

def split_user_timeline(user_df, train_frac=0.7, val_frac=0.15):
    # user_df is sorted by timestamp
    n = len(user_df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_data = user_df.iloc[:train_end]
    val_data = user_df.iloc[train_end:val_end]
    test_data = user_df.iloc[val_end:]
    return train_data, val_data, test_data


# get ready to create negative samples
def get_existing_edges(all_data):
    '''
    gets all the existing edges in the entire dataset

    :param all_data:
    :return:
    '''
    return set(all_data.apply(lambda x: (x['graph_user_id'], x['graph_track_id']), axis = 1))

def get_sorted_tracks_by_timestamp(all_data):
    '''

    :param all_data:
    :return:
    '''
    return all_data[['graph_track_id', 'timestamp']].groupby('graph_track_id').first().sort_values(by='timestamp').reset_index().values

def get_aux_candidates(track_features, track_id_mapping, release_year = 2018):
    return track_features.loc[track_features['release'] < release_year, 'id'].map(track_id_mapping).values


def split_user_timeline(user_df, train_frac=0.7, val_frac=0.15):
    # user_df is sorted by timestamp
    n = len(user_df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_data = user_df.iloc[:train_end]
    val_data = user_df.iloc[train_end:val_end]
    test_data = user_df.iloc[val_end:]
    return train_data, val_data, test_data


# create negative samples for all samples


@njit
def random_sample_set(arr, k=-1):
    n = arr.size
    if n == 0:
        return np.empty(0, dtype=arr.dtype)
    if k < 0 or k > n:
        k = n
    seen = {0}
    seen.clear()
    index = np.empty(k, dtype=np.int64)
    for i in range(k):
        j = random.randint(0, n - 1)
        while j in seen:
            j = random.randint(0, n - 1)
        seen.add(j)
        index[i] = j
    return arr[index]


@njit
def generate_negative_edges_numba(user_ids, track_ids, timestamps, ts_tracks, existing_edges_set, aux_candidate_list):
    positive_edges = []
    negative_edges = []
    ts_tracks_keys = ts_tracks[:, 0]
    ts_tracks_values = ts_tracks[:, 1]
    # user_ids = data['user_id']
    # track_ids = data['track_id']
    # timestamps = data['timestamp']

    for i in range(len(user_ids)):
        user_id = user_ids[i]
        track_id = track_ids[i]
        timestamp = timestamps[i]
        positive_edges.append((user_id, track_id))
        # Get tracks listened to before the current timestamp
        candidates = ts_tracks_keys[ts_tracks_values < timestamp]

        if candidates.size == 0:  # if no valid candidates - choose candidates from previous years
            candidates = aux_candidate_list

        resample = True
        while resample:
            sample = random_sample_set(candidates, k=1)
            sample = sample[0]
            if (user_id, sample) in existing_edges_set:
                continue
            else:
                resample = False
                negative_edges.append((user_id, sample))

    return user_ids, positive_edges, negative_edges


def sample_negative_edges(data, ts_tracks, existing_edges_set, aux_candidates):
    return generate_negative_edges_numba(data['graph_user_id'].values, data['graph_track_id'].values,
                                         data['timestamp'].values, ts_tracks, existing_edges_set, aux_candidates)
