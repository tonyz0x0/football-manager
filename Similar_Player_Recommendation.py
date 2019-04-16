import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import SpectralClustering, KMeans
from sklearn import preprocessing


# import matplotlib.pyplot as plt


def sort_cos_sim(cos_dist, player_index, origin_matrix, x):
    # sort the cosine similarity
    topx = cos_dist[player_index].sort_values(ascending=False)[0:x + 1]
    ids = topx.index
    names = origin_matrix.loc[ids, "Name"]
    return topx, ids, names


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

util_df.index = util_df.loc[:, "ID"]
player_df.index = util_df.loc[:, "ID"]

# ===========Parameter input================

player_rowindex = 50
player_ID = player_df.index[player_rowindex]
sim_counts = 20
print(util_df.loc[player_ID, "Name"])


# ===========similarity matrix==============

scaled_df = preprocessing.scale(player_df)

cosine_dist = pd.DataFrame(cosine_similarity(scaled_df))
cosine_dist.index = util_df.loc[:, "ID"]

topx, similar_style_players_ids2, similar_style_players_names2 = sort_cos_sim(cosine_dist, player_rowindex, util_df,
                                                                              sim_counts)

sim_matrix = np.exp(3 * cosine_dist)
# print(sim_matrix)

# spectral clustering
# clustering = SpectralClustering(n_clusters=50, assign_labels="kmeans", affinity='precomputed')
# clustering.fit(sim_matrix)

# kmeans clustering
clustering = KMeans(n_clusters=50)
clustering.fit(player_df)

# print(clustering.labels_)

cluster_index = clustering.labels_[player_rowindex]
print("Belongs to cluster:", cluster_index)

# get player IDs in the same cluster

similar_style_players_ids1 = []
for i in range(0, player_df.shape[0]):
    if clustering.labels_[i] == cluster_index:
        similar_style_players_ids1.append(util_df.iloc[i].ID)

similar_style_players_matrix1 = pd.DataFrame(player_df.loc[similar_style_players_ids1, :])
similar_style_players_names1 = util_df.loc[similar_style_players_ids1, "Name"]
similar_style_players1_2norm = pd.Series(np.linalg.norm(similar_style_players_matrix1, axis=1))
similar_style_players1_2norm.index = similar_style_players_names1
original_player_2norm_cluster = similar_style_players1_2norm.loc[util_df.loc[player_ID, "Name"]]

sorted_2norm_cluster = similar_style_players1_2norm.sort_values(ascending=False)

for i in range(0, len(similar_style_players1_2norm)):
    if sorted_2norm_cluster[i] > original_player_2norm_cluster:
        print(sorted_2norm_cluster.index[i], "\t", sorted_2norm_cluster[i])
    elif sorted_2norm_cluster[i] == original_player_2norm_cluster:
        print("\nRank below the input player：", original_player_2norm_cluster, "\n")
    else:
        print(sorted_2norm_cluster.index[i], "\t", sorted_2norm_cluster[i])

# euc_dist = euclidean_distances(similar_style_players_matrix1)


print("=============================cosine similarity===================================")

# cosine_sorted_to_2norm

position_player2 = 0

similar_style_players_matrix2 = pd.DataFrame(player_df.loc[similar_style_players_ids2, :])

# euc_dist2 = pd.DataFrame(euclidean_distances(similar_style_players_matrix2))
# euc_dist2.index = similar_style_players_names2

similar_style_players2_2norm = pd.Series(np.linalg.norm(similar_style_players_matrix2, axis=1))
similar_style_players2_2norm.index = similar_style_players_names2
original_player_2norm_cosine = similar_style_players2_2norm[position_player2]

# ================output area===============

print("cluster player count:", len(similar_style_players1_2norm))

print(similar_style_players_names2[0:11], "\n")

# print(euc_dist2[0].sort_values()[1:11])

sorted_2norm = similar_style_players2_2norm.sort_values(ascending=False)
for i in range(0, sim_counts):
    if sorted_2norm[i] > original_player_2norm_cosine:
        print(sorted_2norm.index[i], "\t", sorted_2norm[i])
    elif sorted_2norm[i] == original_player_2norm_cosine:
        print("\nRank below the input player：", original_player_2norm_cosine)
    else:
        print(sorted_2norm.index[i], "\t", sorted_2norm[i])
