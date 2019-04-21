# football-manager

In soccer’ s transfer market, the club managers find it more and more difficult to get a good sign. Therefore, we propose a method to help club managers easily find more competitive players. Our scouting system will enable clubs to increase their scouting scope, decrease their risk of signing the wrong player and enlarge the change of finding the right talent.

## Dataset

### Player Dataset

[Player dataset](https://github.com/BigTony666/football-manager/blob/master/data_clean.csv) originally comes from [sofifa](https://sofifa.com/), includes main features of each individual players, such as height, ball control, long passing, along with performance ratings on different positions and a sum up.

Table structure: $16123\ players * 74\ features$

### Team Dataset

[Team Dataset](https://github.com/BigTony666/football-manager/blob/master/team_feat.csv) includes team features like passing, shooting score and transfer budget.
Table structure: $653\ teams * 30\ features$

## Parallel Weighted Matrix Recommendation

The goal is to calculate every player’s valuable score to every club, represented in a MxN matrix, where M is the number of the players, N is the number of the clubs.

![image](https://user-images.githubusercontent.com/29159357/56473718-120aef80-643d-11e9-9cca-036a8edd749a.png)

### Step 1

Group the player into three main groups: DEF, MID and ATK.

![image](https://user-images.githubusercontent.com/29159357/56473724-29e27380-643d-11e9-858c-f175b266e404.png)

### Step 2

Calculate the Pearson Correlation between each specific position and  specific features, list the 12 most relevant features for each specific positions, group all positions together into only three, ATK, MID and DEF, thus we get the three 1 * 12 vector for three main groups.

![image](https://user-images.githubusercontent.com/29159357/56473784-fb18cd00-643d-11e9-8870-0749912b5c45.png)

### Step 3

Get 3 features-teams matrices parallely(by creating three python processes using Process Pool), transfer the features' score into weights(use Reciprocal Function) and thus we have 3 weighted-teams matrices parallelly.

![image](https://user-images.githubusercontent.com/29159357/56473779-ea685700-643d-11e9-9085-1fd9ba955ec1.png)


### Step 4

Create three sections, each section has a m*n and n*k matrix, where m is the number of players, n is the number of features' weights, and k is the number of teams. For all these three pairs of matrices, do the matrix multiplication. Then we can get 3 MxK matrices for DEF, MID and ATK positions.

![image](https://user-images.githubusercontent.com/29159357/56473788-11268d80-643e-11e9-9c03-7665dea71715.png)

### Step 5

Do the recommendations based on these matrices.

- To recommend Most Valued People in DEF position for FC Barcelona(Or to advise who are the Least Valued People in DEF position for FC Barcelona)

   ![image](https://user-images.githubusercontent.com/29159357/56473791-24395d80-643e-11e9-9f90-187602d6c319.png)

- To find out the Least Valued Players in DEF position in FC Barcelona, thus in the future we can replace them with better players.

   ![image](https://user-images.githubusercontent.com/29159357/56473794-361b0080-643e-11e9-9309-4acfd9817004.png)

## Code Repository

You can find the code here: [football-manager](https://github.com/BigTony666/football-manager), the core method can be found in [recommendation-system.py](https://github.com/BigTony666/football-manager/blob/master/recommendation-system.py).
