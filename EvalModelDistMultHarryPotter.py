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
    modelName="HarryPotterDistMult"
    model=restore_model(model_name_path='saved_model/best_model_DistMult_HarryPotter')
    filter_triples = np.load('filter_triples.npy', allow_pickle=True)
    X_numpy = np.array([
        ['Harry', 'student at', 'Hogwarts School'],
        ['Hermione', 'student at', 'Hogwarts School'],
        ['Ron', 'student at', 'Hogwarts School'],
        ['Voldemort', 'learn', 'details about prophecy'],
        ['Ron', 'destroy', 'Horcrux'],
        ['Harry', 'killed', 'Voldemort'],
        ['Hermione', 'given', 'old potions'],
        ['A ghoul', 'protect', 'Voldemort'],
        ['Harry', 'win', 'Triwizard Cup'],
        ['Hermione', 'are', 'married with three children'],
        ['A ghoul', 'student at', 'Hogwarts School'],
        ['Ron', 'killed', 'her parents'],
        ['Harry friends', 'are relatively protected from', 'Voldemort'],
        ['Ron', 'placed', 'Hogwarts School'],
        ['Hermione', 'develop', 'adolescence'],
        ['Voldemort', 'are', 'married'],
        ['Harry', 'been targeted by', 'Draco'],
        ['Snape', 'killed', 'Harry'],
        ['Hermione', 'develop', 'crush on Cho Chang'],
        ['Hermione', 'manages', 'escape']
    ])

    predicting_links(X_numpy, filter_triples,model, modelName )

    output = pd.read_pickle('saved_model/resultPredictingLinksHarryPotterDistMult')
    print(output)
    threshold_predicting_links('saved_model/resultPredictingLinksHarryPotterDistMult', 10)
    averageMetric('saved_model/resultPredictingLinksHarryPotterDistMultThresholded.pkl')