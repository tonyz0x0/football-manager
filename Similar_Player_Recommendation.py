"""

Module: Similar_Player_Recommendation
Author: Jiajun Sheng
Usage: spr(player_index='', player_id='', ncluster=9, sim_counts=20, method="spectral")
        method = ["spectral", "kmeans", "cosine_sim"]
        player_id is ignored if player_index is provided
Examples:
    from Similar_Player_Recommendation import spr

    spr(player_index=50, method="spectral")
    spr(player_id=190483, sim_counts=15, method="cosine_sim")
    spr(player_index=123, ncluster=8, method="kmeans")

"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import SpectralClustering, KMeans
from sklearn import preprocessing
# from sklearn import metrics
# import matplotlib.pyplot as plt


def sort_cos_sim(cos_dist, player_index, origin_matrix, x):
    # sort the cosine similarity
    topx = cos_dist[player_index].sort_values(ascending=False)[0:x + 1]
    ids = topx.index
    names = origin_matrix.loc[ids, "Name"]
    return topx, ids, names


def spr(player_index='', player_id='', ncluster=9, sim_counts=20, method="spectral"):

    # ==============matrix building================

    util_df = pd.read_csv('data_clean.csv')
    # print(util_df.info())

    characteristics = ['Crossing', 'Finishing', 'HeadingAccuracy',
                       'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
                       'FKAccuracy', 'LongPassing', 'BallControl',
                       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions',
                       'Balance', 'ShotPower', 'Jumping', 'Stamina',
                       'Strength', 'LongShots', 'Aggression',
                       'Interceptions', 'Positioning', 'Vision',
                       'Penalties', 'Composure', 'Marking', 'StandingTackle',
                       'SlidingTackle']
    player_df = util_df.loc[:, characteristics]
    # print(player_df.info())

    index_id = util_df.loc[:, "ID"]
    # print(index_id)
    # return

    util_df.index = util_df.loc[:, "ID"]
    player_df.index = util_df.loc[:, "ID"]

    # ===========Parameter input================
    if not player_index:
        player_rowindex = pd.array(index_id[index_id == player_id].index)[0]
    else:
        player_rowindex = player_index
        player_id = player_df.index[player_rowindex]
    n = ncluster
    print(util_df.loc[player_id, "Name"])

    # ===========similarity matrix==============

    scaled_df = preprocessing.scale(player_df)

    cosine_dist = pd.DataFrame(cosine_similarity(scaled_df))
    cosine_dist.index = util_df.loc[:, "ID"]

    sim_matrix = np.exp(3 * cosine_dist)
    # print(sim_matrix)

    # ======================clustering=====================
    global clustering
    if method == "spectral":
        # spectral clustering
        print("Method = Spectral clustering")
        clustering = SpectralClustering(n_clusters=n, assign_labels="discretize", affinity='precomputed')
        clustering.fit(sim_matrix)

    elif method == "kmeans":
        # kmeans clustering
        print("Method = Kmeans")
        clustering = KMeans(n_clusters=n)
        clustering.fit(player_df)

    elif method == "cosine_sim":
        print("=============================cosine similarity===================================")

        # sort most similar topx players_matrix, ids, and names
        topx, similar_style_players_ids2, similar_style_players_names2 = sort_cos_sim(cosine_dist, player_rowindex,
                                                                                      util_df,
                                                                                      sim_counts)
        # cosine_sorted_to_2norm
        position_player2 = 0
        similar_style_players_matrix2 = pd.DataFrame(player_df.loc[similar_style_players_ids2, :])

        # euc_dist2 = pd.DataFrame(euclidean_distances(similar_style_players_matrix2))
        # euc_dist2.index = similar_style_players_names2

        similar_style_players2_2norm = pd.Series(np.linalg.norm(similar_style_players_matrix2, axis=1))
        similar_style_players2_2norm.index = similar_style_players_names2
        original_player_2norm_cosine = similar_style_players2_2norm[position_player2]

        print("Most similar players:\n", similar_style_players_names2[0:11], "\n")

        # print(euc_dist2[0].sort_values()[1:11])

        # output cosine
        print("Recommendation:")
        sorted_2norm = similar_style_players2_2norm.sort_values(ascending=False)
        for i in range(0, sim_counts):
            if sorted_2norm[i] > original_player_2norm_cosine:
                print(sorted_2norm.index[i], "\t", sorted_2norm[i])
            elif sorted_2norm[i] == original_player_2norm_cosine:
                print("\nRank below the input player：", original_player_2norm_cosine)
            else:
                print(sorted_2norm.index[i], "\t", sorted_2norm[i])
        return
    else:
        exit("method_err")

        # sil_score = metrics.silhouette_score(player_df, clustering.labels_, metric='cosine')
        # print(n, sil_score)
        # cal_score = metrics.calinski_harabaz_score(player_df, clustering.labels_)
        # print(n, "\t", cal_score)
        # print(clustering.labels_)

    cluster_index = clustering.labels_[player_rowindex]
    # print("Belongs to cluster:", cluster_index)

    # get player IDs in the same cluster
    similar_style_players_ids1 = []
    for i in range(0, len(clustering.labels_)):
        if clustering.labels_[i] == cluster_index:
            similar_style_players_ids1.append(util_df.iloc[i].ID)

    similar_style_players_matrix1 = pd.DataFrame(player_df.loc[similar_style_players_ids1, :])
    similar_style_players_names1 = util_df.loc[similar_style_players_ids1, "Name"]
    similar_style_players1_2norm = pd.Series(np.linalg.norm(similar_style_players_matrix1, axis=1))
    similar_style_players1_2norm.index = similar_style_players_names1
    original_player_2norm_cluster = similar_style_players1_2norm.loc[util_df.loc[player_id, "Name"]]

    sorted_2norm_cluster = similar_style_players1_2norm.sort_values(ascending=False)

    for i in range(0, min(len(similar_style_players1_2norm), 10)):
        if sorted_2norm_cluster[i] > original_player_2norm_cluster:
            print(sorted_2norm_cluster.index[i], "\t", sorted_2norm_cluster[i])
        elif sorted_2norm_cluster[i] == original_player_2norm_cluster:
            print("\nRank below the input player：", original_player_2norm_cluster, "\n")
        else:
            print(sorted_2norm_cluster.index[i], "\t", sorted_2norm_cluster[i])

    # euc_dist = euclidean_distances(similar_style_players_matrix1)

    # ================output area===============

    print("cluster player count:", len(similar_style_players1_2norm))


