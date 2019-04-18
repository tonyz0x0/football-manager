#!/usr/bin/env python
# coding: utf-8

# In[280]:


from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml.stat import Correlation
from pyspark.ml.linalg import Vectors
import numpy as np
import pandas as pd
from collections import Counter
import re
import time

spark = SparkSession.builder.appName('FootballManager').getOrCreate()


# In[261]:


def readCSV(path):
    return spark.read.format("csv").options(header="true", inferSchema="true")    .load(path)

teams_df = readCSV("./team_feat.csv")
players_df = readCSV("./data_clean.csv")
_teams_df = pd.read_csv('./team_feat.csv')
_players_df = pd.read_csv('./data_clean.csv')


# In[290]:


class RecommendationEngine(object):
    def __init__(self, players_df, teams_df, _players_df, _teams_df):
        self.players_df = players_df
        self.teams_df = teams_df
        self._teams_df = _teams_df
        self._players_df = _players_df
        self.result = {}
    
    def readCSVToSparksql(self, path):
        return spark.read.format("csv").options(header="true", inferSchema="true")            .load(path)

    def __convertDFtoRDD(self, df):
        rdd = df.rdd
        return rdd
    
#     def __groupPosition(self, players_df):
#         """
#         Function to group the players' positions into three main groups: DEF, MID, FWD
        
#         :param: players_df
#         :return: Dataframe
#         """
        
#         def _classify(position):
#             """
#             Classify Position
            
#             :param: position
#             :return: string
#             """
#             # Regex to group
#             defs = r'\w*B$'
#             mids = r'\w*M$'
#             fwds = r'\w*[FSTW]$'
            
#             if re.match(defs, position):
#                 return "DEF"
#             elif re.match(mids, position):
#                 return "MID"
#             elif re.match(fwds, position):
#                 return "FWD"
#             else:
#                 return None
        
#         # Write an UDF for withColumn
#         _classify_udf = udf(_classify, StringType())
        
#         # groupPosition list
#         return players_df\
#           .withColumn('GroupPosition', _classify_udf(players_df['Position']))

    def __groupPosition(self, players_df):
        # Regex to group
        defs = r'\w*B$'
        mids = r'\w*M$'
        fwds = r'\w*[FSTW]$'

        # groupPosition list
        groupPositions = []
        for index, row in players_df.iterrows():
            position = row['Position']
            if re.match(defs, position):
                groupPositions.append('DEF')
            if re.match(mids, position):
                groupPositions.append('MID')
            if re.match(fwds, position):
                groupPositions.append('FWD')
        series = pd.Series(groupPositions)
        players_df['GroupPosition'] = series
        return players_df
        
    def __findTopRelatedPosition(self, players_df):
        """
        Calculate the Pearson Correlation between each specific position,
        and specific featuresFind out top characteristics for different position
        
        :param: players_df, Pandas Dataframe
        :return: 
        """
        player_characteristics = ['Crossing','Finishing', 'HeadingAccuracy', 
                                  'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
                                  'FKAccuracy', 'LongPassing', 'BallControl', 
                                  'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 
                                  'Balance', 'ShotPower', 'Jumping', 'Stamina', 
                                  'Strength', 'LongShots', 'Aggression',
                                  'Interceptions', 'Positioning', 'Vision', 
                                  'Penalties', 'Composure', 'Marking', 'StandingTackle', 
                                  'SlidingTackle']

        ## Top characteristics for  positions
        corr_matrix = players_df.corr() # default is pearson
        counter_DEF = Counter()
        counter_MID = Counter()
        counter_ATK = Counter()
        defs = r'\w*B$'
        mids = r'\w*M$'
        fwds = r'\w*[FSTW]$'
        for index, row in corr_matrix.loc[player_characteristics, "LS":"RB"].T.iterrows():

            largests = tuple(row.nlargest(12).index)
        #     print('Position {}: {}, {}, {}, {}, {}, {}, {}, {}'.format(index, *tuple(row.nlargest(8).index)))

            if re.match(defs, index): # DEF
                for feature in largests:
                    counter_DEF[feature] += 1
            if re.match(mids, index): # MID
                for feature in largests:
                    counter_MID[feature] += 1
            if re.match(fwds, index): # FWD
                for feature in largests:
                    counter_ATK[feature] += 1

        # 3. Group all positions together into only three, ATK, MID and DEF, thus we get the three 1 * 8 vector for three main groups.
        features_DEF = [kv[0] for kv in counter_DEF.most_common(12)]
        features_MID = [kv[0] for kv in counter_MID.most_common(12)]
        features_ATK = [kv[0] for kv in counter_ATK.most_common(12)]
        
        return features_DEF, features_MID, features_ATK
    
    def _getWeightedMatrix(self, teams_df, input):
        """
        Get 3 features-teams matrixs , transfer the features' score into weights(use Reciprocal Function) and thus we have 3 weighted-teams matrixs.
        
        :param: input, a dic of features vectors
        :return a tuple of weighted-teams matrixs
        """
        features_DEF = input['features_DEF']
        features_MID = input['features_MID']
        features_ATK = input['features_ATK']
                
        teams_columns = dict(zip(range(0,teams_df['Club'].size), teams_df['Club']))

        features_teams_DEF = teams_df.loc[:, features_DEF].T
        features_teams_DEF = features_teams_DEF.rename(columns=teams_columns)
        features_teams_MID = teams_df.loc[:, features_MID].T
        features_teams_MID = features_teams_MID.rename(columns=teams_columns)
        features_teams_ATK = teams_df.loc[:, features_ATK].T
        features_teams_ATK = features_teams_ATK.rename(columns=teams_columns)
        
        weights_teams_DEF = features_teams_DEF.applymap(lambda x: 1./float(x))
        weights_teams_MID = features_teams_MID.applymap(lambda x: 1./float(x))
        weights_teams_ATK = features_teams_ATK.applymap(lambda x: 1./float(x))
        
        return weights_teams_DEF, weights_teams_MID, weights_teams_ATK
        
    def _getPlayersTeamsMatrix(self, players_df, teams_df, input):
        """
        Create three sections, each section has a m*n and n*k matrix,
        where m is the number of players, n is the number of features' weights,
        and k is the number of teams. For all these three pairs of matrices,
        do the matrix multiplication. Then we can get 3 MxK matrices for DEF, MID and ATK positions.
        
        :params: input, a dict 
        : return: a tuple of three players_teams matrixs
        """
        
        features_DEF = input['features_DEF']
        features_MID = input['features_MID']
        features_ATK = input['features_ATK']
        weights_teams_DEF = input['weights_teams_DEF']
        weights_teams_MID = input['weights_teams_MID']
        weights_teams_ATK = input['weights_teams_ATK']
        
        players_rows = dict(zip(range(0, players_df['Name'].size), players_df['Name']))

        # DEF
        players_features_DEF = players_df.loc[:, features_DEF]
        players_features_DEF = players_features_DEF.rename(index=players_rows)
        players_teams_DEF = players_features_DEF.dot(weights_teams_DEF)

        # MID
        players_features_MID = players_df.loc[:, features_MID]
        players_features_MID = players_features_MID.rename(index=players_rows)
        players_teams_MID = players_features_MID.dot(weights_teams_MID)

        # ATK
        players_features_ATK = players_df.loc[:, features_ATK]
        players_features_ATK = players_features_ATK.rename(index=players_rows)
        players_teams_ATK = players_features_ATK.dot(weights_teams_ATK)
        
        return players_teams_DEF, players_teams_MID, players_teams_ATK

    def _run(self):
        """
        Run the recommendation engine
        
        :return: a tuple of three players_teams matrixs
        """
        start=time.time()
        self._players_df = self.__groupPosition(self._players_df)
        features_DEF, features_MID, features_ATK = self.__findTopRelatedPosition(self._players_df)
        weights_teams_DEF, weights_teams_MID, weights_teams_ATK = self._getWeightedMatrix(
            self._teams_df,
            {'features_DEF': features_DEF,
             'features_MID': features_MID,
             'features_ATK': features_ATK
            }
        )
        players_teams_DEF, players_teams_MID, players_teams_ATK = self._getPlayersTeamsMatrix(
            self._players_df,
            self._teams_df,
            {'features_DEF': features_DEF,
             'features_MID': features_MID,
             'features_ATK': features_ATK,
             'weights_teams_DEF': weights_teams_DEF,
             'weights_teams_MID': weights_teams_MID,
             'weights_teams_ATK': weights_teams_ATK
            }
        )
        end=time.time()
        print("The recommendation running time is: {:.2f} seconds".format(end-start))
        return players_teams_DEF, players_teams_MID, players_teams_ATK
    
    def getRecommendation(self):
        """
        Get the recommendation result, use lazy load mode
        
        :return: a tuple of three players_teams matrixs
        """
        if len(self.result) == 0:
            players_teams_DEF, players_teams_MID, players_teams_ATK = self._run()
            self.result['players_teams_DEF'] = players_teams_DEF
            self.result['players_teams_MID'] = players_teams_MID
            self.result['players_teams_ATK'] = players_teams_ATK
        return self.result
    

class RecommendationSystem(object):
    def __init__(self, recommendation_engine, players_df, teams_df, _players_df, _teams_df):
        self.players_df = players_df
        self.teams_df = teams_df
        self._teams_df = _teams_df
        self._players_df = _players_df
        self._recommendation_engine = recommendation_engine
        self.players_teams_matrixs = self._recommendation_engine.getRecommendation()
    
    def getMVPForTeam(self, team, position, K, isReverse=False):
        """
        To recommend K Most Valued People in specific position for specific team if ascending is False,
        otherwise show Worst Valued People if ascending is True
        
        :param: team
        :param: position
        :param: K
        :param: isReverse, default False
        :return: players -> List[]
        """
        if position == 'DEF':
            players = self.players_teams_matrixs['players_teams_DEF'][team]                       .sort_values(ascending=isReverse)                       .head(K)
            return players
        elif position == 'MID':
            players = self.players_teams_matrixs['players_teams_MID'][team]                       .sort_values(ascending=isReverse)                       .head(K)
            return players
        elif position == 'ATK':
            players = self.players_teams_matrixs['players_teams_ATK'][team]                       .sort_values(ascending=isReverse)                       .head(K)
            return players
        else:
            raise RuntimeError('Invalid position argument')
    
    def getMVTForPlayer(self, player, position, K, isReverse=False):
        """
        To recommend Most Valued Teams in specific position for specific palyer if ascending is False,
        otherwise show Worst Valued Teams if ascending is True
        
        :param: player
        :param: position
        :param: K
        :param: isReverse, default False
        :return: teams -> List[]
        """
        if position == 'DEF':
            teams = self.players_teams_matrixs['players_teams_DEF']                       .loc[player, :]                       .sort_values(ascending=isReverse)                       .head(K)
            return teams
        elif position == 'MID':
            teams = self.players_teams_matrixs['players_teams_MID']                       .loc[player, :]                       .sort_values(ascending=isReverse)                       .head(K)
            return teams
        elif position == 'ATK':
            teams = self.players_teams_matrixs['players_teams_ATK']                       .loc[player, :]                       .sort_values(ascending=isReverse)                       .head(K)
            return teams
        else:
            raise RuntimeError('Invalid position argument')
    
    def searchWorstPlayersInPosByTeam(self, position, team):
        """
        To find out the Least Valued Players in specific position for specific team,
        thus in the future we can replace them with better players.
        
        :params: position
        :params: team
        :return:
        """
        players_df = self._players_df
        players_teams_matrix = None
        if position == 'DEF':
            players_teams_matrix = self.players_teams_matrixs['players_teams_DEF']
        elif position == 'MID':
            players_teams_matrix = self.players_teams_matrixs['players_teams_MID']
        elif position == 'ATK':
            players_teams_matrix = self.players_teams_matrixs['players_teams_ATK']
        else:
            raise RuntimeError('Invalid position argument')
        for index, value in players_teams_matrix[team].sort_values(ascending=True).iteritems():
            if players_df.loc[(players_df.loc[:, 'Name'] == index), :]['Club'].values[0] == team and players_df.loc[(players_df.loc[:, 'Name'] == index), :]['GroupPosition'].values[0] == position:
                print("{}\t\t{}".format(index, value))
        
class FootballManager(object):
    def __init__(self, recommendation_system):
        self.players_df = self.readCSVToSparksql("./data_clean.csv")
        self.teams_df = self.readCSVToSparksql("./team_feat.csv")
        self._teams_df = pd.read_csv('./team_feat.csv')
        self._players_df = pd.read_csv('./data_clean.csv')
        self.recommendation_system = recommendation_system
    
    def readCSVToSparksql(self, path):
        return spark.read.format("csv").options(header="true", inferSchema="true")            .load(path)

    def matrix_weighted_recommandation(self, input_player=None, input_team=None):
        if input_player is None and input_team is None:
            raise RuntimeError('No player or team is found, please at least offer one argument')
        

    


# In[291]:


recommendation_engine = RecommendationEngine(players_df, teams_df, _players_df, _teams_df)


# In[292]:


recommendationSystem = RecommendationSystem(recommendation_engine, players_df, teams_df, _players_df, _teams_df)


# In[293]:


players = recommendationSystem.getMVPForTeam('LA Galaxy', 'DEF', 10)
print(players)
teams = recommendationSystem.getMVTForPlayer('L. Messi', 'DEF', 10)
print(teams)
recommendationSystem.searchWorstPlayersInPosByTeam('DEF', 'LA Galaxy')

