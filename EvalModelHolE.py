import numpy as np
import pandas as pd
from ampligraph.utils import restore_model
from PredictingLinks import predicting_links


def threshold_predicting_links(pathToDataframe, numberOfThresholded):
    dataframe = pd.read_pickle(pathToDataframe)
    df = dataframe.iloc[:numberOfThresholded, ]
    print(df)
    df.to_pickle(pathToDataframe + "Thresholded.pkl")

def averageMetric(pathToDataFrame):
    list=[]
    dataframe=pd.read_pickle(pathToDataFrame)
    for column in dataframe['rank']:
        print(column)
        list.append(column)
    mean= np.mean(np.array(list))
    print(mean)
    return mean


if __name__ == '__main__':
    modelName="HolE"
    model=restore_model(model_name_path='saved_model/best_model_HolE')

    filter_triples=np.load('filter_triples.npy', allow_pickle=True)
    X_numpy= np.array([
        ['Jorah Mormont', 'SPOUSE', 'Daenerys Targaryen'],
        ['Tyrion Lannister', 'SPOUSE', 'Missandei'],
        ["King's Landing", 'SEAT_OF', 'House Lannister of Casterly Rock'],
        ['Sansa Stark', 'SPOUSE', 'Petyr Baelish'],
        ['Daenerys Targaryen', 'SPOUSE', 'Jon Snow'],
        ['Daenerys Targaryen', 'SPOUSE', 'Craster'],
        ['House Stark of Winterfell', 'IN_REGION', 'The North'],
        ['House Stark of Winterfell', 'IN_REGION', 'Dorne'],
        ['House Tyrell of Highgarden', 'IN_REGION', 'Beyond the Wall'],
        ['Brandon Stark', 'ALLIED_WITH', 'House Stark of Winterfell'],
        ['Brandon Stark', 'ALLIED_WITH', 'House Lannister of Casterly Rock'],
        ['Rhaegar Targaryen', 'PARENT_OF', 'Jon Snow'],
        ['House Hutcheson', 'SWORN_TO', 'House Tyrell of Highgarden'],
        ['Daenerys Targaryen', 'ALLIED_WITH', 'House Stark of Winterfell'],
        ['Daenerys Targaryen', 'ALLIED_WITH', 'House Lannister of Casterly Rock'],
        ['Jaime Lannister', 'PARENT_OF', 'Myrcella Baratheon'],
        ['Robert I Baratheon', 'PARENT_OF', 'Myrcella Baratheon'],
        ['Cersei Lannister', 'PARENT_OF', 'Myrcella Baratheon'],
        ['Cersei Lannister', 'PARENT_OF', 'Brandon Stark'],
        ["Tywin Lannister", 'PARENT_OF', 'Jaime Lannister'],
        ["Missandei", 'SPOUSE', 'Grey Worm'],
        ["Brienne of Tarth", 'SPOUSE', 'Jaime Lannister']
    ])

    predicting_links(X_numpy, filter_triples,model, modelName )

    output=pd.read_pickle('saved_model/resultPredictingLinksHolE')
    print(output)
    threshold_predicting_links('saved_model/resultPredictingLinksHolE', 10)
    averageMetric('saved_model/resultPredictingLinksHolEThresholded.pkl')