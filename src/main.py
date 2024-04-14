import pandas as pd
from  RISE import RISE
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    pathtodataset = input("please input the absolute path to the csv")
    dataset = pd.DataFrame(pd.read_csv(pathtodataset))
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    
    ruleset = RISE.train(dataset)
    