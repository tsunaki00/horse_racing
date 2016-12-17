#coding:utf-8
import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class BestParameter:
  def __init__ (self) :
    self.horse_data = []  
    self.train_data = []  
    self.train_target = [] 
    # テスト対象
    self.test_row_no = -1 

    self.master = {
      1 : {
        "福島": 0, "小倉": 1, "京都": 2, "函館": 3,
        "中山": 4, "札幌": 5, "東京": 6,
        "阪神": 7, "中京": 8, "新潟": 9 
      },
      4 :  { "芝" : 0, "ダート" : 1, "障害" : 2 },
      5 :  { "右" : 0, "左" : 1, "芝" : 2, "直線" : 3, "右2周" : 4 },
      7 :  { "不良" : 0,  "重" : 1, "稍重" : 2, "良" : 3 },
      14 : {"牡" : 0, "牝" : 1, "せん" : 2},
      29 : {"A" : 0, "B" : 1, "C" : 2, "D" : 3, "E" : 4, "nan" : -1 }
    }

  def best_parameter(self):
    hurdle_race_count = 0
    header = []
    label = []
    with open("data/jra_race_result.csv", "r") as f:
      reader = csv.reader(f)
      # 障害は除くデータで予測データを作成
      for idx, row in enumerate(reader):
        if idx == 0:
          for i, col in enumerate(row):
            header = row
          continue
        elif row[4] == '障害' :
          hurdle_race_count += 1
          continue
        horse = []
        parameter = []
        # マスタデータで数値化
        for i, col in enumerate(row):
          if i in {3, 13, 16, 18, 19, 26, 27, 28}:
            horse.append(col)
            continue
          elif i == 0 : 
            if self.test_row_no == -1 and col == '2016-09-17' :
              self.test_row_no = (idx - hurdle_race_count)
            parameter.append(col.replace('-',''))
          elif i == 10 : 
            label.append(header[i])
            horse.append(col)
            self.train_target.append(col)
          elif self.master.has_key(i) :
            if i == 1 :
              horse.append(col)
            label.append(header[i])
            parameter.append(self.master[i][col])
          else :
            if i in (2, 12) :
              horse.append(col)
            label.append(header[i])
            if col == ''  or col == ' - ': 
              col = -1
            parameter.append(float(col))
        self.horse_data.append(horse)
        self.train_data.append(parameter)
    
    # fitで学習 (9/17までを学習)
    # modelをシリアライズする場合
    # joblib.dump(model, 'model.pkl') 
    # 素性の重要度（RandomForestの分岐での重要度）
    parameters = {
      'n_estimators'      : [5, 10],
      'max_features'      : ['auto', 'sqrt', 'log2', None],
      'max_features'      : [3, 5, 10, 15, 20],
      'random_state'      : [0],
      'n_jobs'            : [1],
      'min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50],
      'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 40, 50]
    }
    model = GridSearchCV(RandomForestClassifier(), parameters, n_jobs=-1)
    model.fit(self.train_data[0 : self.test_row_no - 1], self.train_target[0 : self.test_row_no - 1])

    for params, mean_score, all_scores in model.grid_scores_:
      print("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

    ## best_estimator_でscoreを表示します。
    print(model.best_estimator_)

if __name__ == "__main__":
  best_parameter = BestParameter()
  best_parameter.best_parameter()



