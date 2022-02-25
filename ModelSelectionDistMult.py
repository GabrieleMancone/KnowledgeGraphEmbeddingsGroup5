import numpy as np
from ampligraph.utils import save_model
from ampligraph.datasets import load_from_csv
from ampligraph.latent_features import DistMult
from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.evaluation import select_best_model_ranking


class ModelSelectionDistMult:

    def __init__(self, arrayBatches, arrayEpochs, arrayK, arrayEta,arrayModelParam, arrayLoss, arrayRegulizer, arrayRegulizerParam, arrayOptimizer, arrayOptimizerParam):
        self.modelName=DistMult
        self.modelParam_grid={
            "seed": [0],
            "batches_count": arrayBatches,
            "epochs": arrayEpochs,
            "k":arrayK,
            "eta": arrayEta,
            "embedding_model_params": arrayModelParam,
            "loss": arrayLoss,
            "regularizer": arrayRegulizer,
            "regularizer_params":arrayRegulizerParam,
            "optimizer": arrayOptimizer,
            "optimizer_params": arrayOptimizerParam,
            "verbose": True
        }

    def fit_eval(self, X_train,  X_val, X_test):
        best_model, best_params, best_mrr_train, ranks_test, mrr_test , experimental_history=select_best_model_ranking(self.modelName,
                                                            X_train,
                                                            X_val,
                                                            X_test,
                                                            self.modelParam_grid,
                                                            use_default_protocol=True,
                                                            early_stopping=True,
                                                            use_filter=True,
                                                            verbose=True)
        return best_model, best_params, best_mrr_train, ranks_test, mrr_test, experimental_history


if __name__ == '__main__':
    X=load_from_csv('.', 'GoT.csv', sep=',')
    num_test=int(len(X) * (20 / 100))
    data={}
    data['X_train_valid'], data['X_test']=train_test_split_no_unseen(X, test_size=num_test, seed=0, allow_duplication=False)
    data['X_train'], data['X_valid']=train_test_split_no_unseen(data['X_train_valid'], test_size=num_test, seed=0, allow_duplication=False)
    filter_triples=np.concatenate((data['X_train'], data['X_valid'], data['X_test']))

    with open('filter_triples.npy', 'wb') as f:
        np.save(f, filter_triples)

    model=ModelSelectionDistMult([10], [500], [200, 150], [10,15],
                                 { "negative_corruption_entities": ["all" , "batch"], "corrupt_sides": 's+o', "normalize_ent_emb": False},
                                 ["pairwise","nll", "self_adversarial","multiclass_nll", "absolute_margin"],
                                 ["LP"],
                                 {"p":[2, 3],"lambda": [1e-4, 1e-5]},
                                 ["adam", "momentum", "sgd", "adagrad"],
                                 {"lr": [0.01, 0.001, 0.0001], "momentum": [ 0.6, 0.7, 0.9]},
                                )

    best_model, best_params, best_mrr_train, ranks_test, mrr_test, experimental_history=model.fit_eval(data['X_train'], data['X_valid'],data['X_test'])
    print(type(best_model).__name__, best_params, best_mrr_train, mrr_test, experimental_history)
    save_model(best_model, model_name_path='saved_model/best_model'+ "{model.modelName}")