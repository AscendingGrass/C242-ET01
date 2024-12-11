import pandas as pd
import json

train_csv_path = 'dataset/train.csv'
test_csv_path = 'dataset/test.csv'

dataset_train = pd.DataFrame(columns=['context', 'questions', 'difficulty'])
dataset_test = pd.DataFrame(columns=['context', 'questions', 'difficulty'])

prefixes = ["easy", "medium", "hard"]
suffixes = range(10, 71, 10)

for difficulty in prefixes:
    for suffix in suffixes:
        with open(f"dataset/raw/{difficulty}_{suffix}.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if suffix == 10:
            dataframe = dataset_test
        else:
            dataframe = dataset_train

        for datum in data:
            context = datum['context']
            questions =  json.dumps(datum['questions'])
            dataframe.loc[len(dataframe)] = [context, questions, difficulty]

dataset_train.to_csv(train_csv_path, index=False)
dataset_test.to_csv(test_csv_path, index=False)