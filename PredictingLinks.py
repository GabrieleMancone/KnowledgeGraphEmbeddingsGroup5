import numpy as np
import pandas as pd
from scipy.special import expit
from ampligraph.evaluation import evaluate_performance

def predicting_links(unseen_triple, filter, model, modelName):

    unseen_filter=np.array(list({tuple(i) for i in np.vstack((filter, unseen_triple))}))

    ranks_unseen=evaluate_performance(
        unseen_triple,
        model=model,
        filter_triples=unseen_filter,
        filter_unseen=True,
        corrupt_side='s+o',
        use_default_protocol=False,
        verbose=True
    )

    scores=model.predict(unseen_triple)
    probs=expit(scores)

    result=pd.DataFrame(list(zip([' '.join(x) for x in unseen_triple],
                        ranks_unseen,
                        np.squeeze(scores),
                        np.squeeze(probs))),
                        columns=['statement', 'rank', 'score', 'prob']).sort_values("rank")

    result.to_pickle("saved_model/resultPredictingLinks"+ modelName)