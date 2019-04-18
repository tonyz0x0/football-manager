#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StringType
import time
import numpy as np

sc =SparkContext.getOrCreate()
sqlContext = SQLContext(sc)


# In[3]:


def readCSV(path):
    '''
    read csv file
    return spark sql data frame
    '''
    return sqlContext.read.format("csv").options(header="true")    .load(path)


# In[4]:


team_df = readCSV("/Users/peggy/Desktop/footballManager/team_feat.csv")


# In[5]:



player_df = readCSV("/Users/peggy/Desktop/footballManager/data_clean.csv")


# In[6]:


def playerSimilarity(p1, p2):
    '''
    length of p2 times cosine of p1 & p2
    '''
    cosine = np.dot(p1,p2)/(np.linalg.norm(p1)*(np.linalg.norm(p2)))
    r =  np.sqrt(sum([i ** 2 for i in p1]))
    return r * cosine


# In[7]:


def findTopK(playerList, K, player, sort_type):
    playerList.append(player)
    playerList.sort(key=lambda p: sort_type * p[1])
    if(len(playerList) > K):
        return playerList[:K]
    return playerList


def mergeTopK(pList1, pList2, K, sort_type):
    result = pList1 + pList2
    result.sort(key=lambda p:sort_type*p[1])
    if(len(result) > K):
        return result[:K]
    return result


# In[22]:


def findSimilarPlayer(df, name, topK):
    '''
    given dataset and target player name
    return top K most similar players data frame of target player
    '''
    player_df = df.select(["ID"] + df.columns[44:73]).where(df.Name == name)
    if player_df == None:
        raise NameError("No Player Found!")
    playerInfo = player_df.rdd.map(list)        .map(lambda l:(l[0], [int(l[i]) for i in range(1, len(l))])).collect()[0]
    (playerId, playerList) = playerInfo[0], playerInfo[1]
    mat = df.select(["ID"] + df.columns[44:73]).rdd.map(list)        .map(lambda l:(l[0], [int(l[i]) for i in range(1, len(l))]))        .filter(lambda kv: kv[0] != playerId)        .mapValues(lambda l: playerSimilarity(l, playerList))

    res = mat.aggregate([], lambda inp1, inp2: findTopK(inp1, topK, inp2, -1), lambda inp1, inp2: mergeTopK(inp1, inp2, topK, -1))
    res = [id for id, score in res]
    id_df = sqlContext.createDataFrame(res, StringType()).toDF("ID")
    res = df.join(id_df, "ID", "inner").select("Name", "Age", "Nationality", "Club", "Height(cm)", "Weight(lbs)")
    return res
    
time1 = time.time()
findSimilarPlayer(player_df, "L. Messi", 10)
run_time = time.time() - time1
print("run time: " + str(run_time))
    


# In[17]:


def findBestReplicate(teamName, playerId, df, topK, weightVector):
    '''
    return list of [(player_id, replace_id, improve score)]
    '''
    player_info = df.select(df.columns[44:73]).where(df.ID == playerId).rdd.map(list)            .map(lambda l: [float(i) for i in l]).collect()[0] # list
    candidatePlayers = df.select(["ID"] + df.columns[44:73]).where(df.Club != teamName).rdd.map(list)        .map(lambda l:(l[0], [float(l[i]) for i in range(1, len(l))]))        .mapValues(lambda vals: improve(vals, player_info, weightVector)) # rdd
    res = candidatePlayers.aggregate([], lambda inp1, inp2: findTopK(inp1, topK, inp2, -1), lambda inp1, inp2: mergeTopK(inp1, inp2, topK, -1))
    res = [(playerId, id, score) for id, score in res]
    return res

def improve(l1, l2, weight):
    improve = 0
    for i in range(len(l1)):
        improve += (l1[i] - l2[i]) * weight[i]
    return improve


# In[21]:


def featureThreshold(l):
    temp = sorted(l)
    return temp[int(len(l) / 4)]


def findWorstFeatures(teamName, team_df):
    '''
    take the team name and team dataframe and return list of index of weak features start from 0 = Crossing
    '''
    targ_df = team_df.select('*').where(team_df.Club == teamName).rdd.map(list)            .map(lambda l: (l[0], [float(l[i]) for i in range(1, len(l))]))            .mapValues(lambda l: (featureThreshold(l), l))            .mapValues(lambda tup: [index for index, val in enumerate(tup[1]) if val < tup[0]])
    feature_indexes = targ_df.collect()[0][1]
    return feature_indexes
    
    
def createWeightVector(feature_indexes):
    '''
    take list of weak features and return weight list of size 29
    '''
    norm = float(10 / (29 + len(feature_indexes)))
    weightVector = [2.0 * norm if index in feature_indexes else norm for index in range(29)]
    return weightVector
     
    
def findWorstPlayers(teamName, player_df, feature_indexes):
    '''
    take team name, player dataframe, weak features index list
    return list of worst players id
    '''
    worst_players = player_df.select(["ID"] + player_df.columns[44:73]).where(player_df.Club == teamName).rdd.map(list)            .map(lambda l: (l[0], [float(i) for i in l[1:]]))            .mapValues(lambda l: [l[i] for i in range(len(l)) if i in feature_indexes])            .mapValues(lambda l: sum(l)).collect()
    worst_players.sort(key = lambda t: t[1], reverse=True)
    return [id for id, index in worst_players][:10]


    
def replaceModeRecommendation(player_df, team_df, teamName, topK):
    feature_indexes = findWorstFeatures(teamName, team_df)
#     print([team_df.columns[i + 1] for i in feature_indexes])
    weight_vector = createWeightVector(feature_indexes)
#     print(weight_vector)
    worst_players = findWorstPlayers(teamName, player_df, feature_indexes)
    res = []
    for player_id in worst_players:
        res += findBestReplicate(teamName, player_id, player_df, topK, weight_vector)
    res.sort(key = lambda l: l[2], reverse=True)
    return res[:topK]
    


def printPlayerInfo(player_df, playerId):
    player_info = player_df.select("ID", 'Name', "Age", "Nationality", "Overall", "Club", "Position")            .where(player_df.ID == playerId).show()



# team_name = 'FC Barcelona'
time1 = time.time()
team_name = 'LA Galaxy'
res = replaceModeRecommendation(player_df, team_df, team_name, 3)
print("run time: " + str(time.time() - time1))
# for i in res:
#     print("player:" + i[0] +" replacement:" + i[1] + " improvement:" + str(i[2]))


# In[ ]:





# In[ ]:




