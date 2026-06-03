import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier


def decision_tree_per_label(X, y, depth):
    tree_model = MultiOutputClassifier(DecisionTreeClassifier(max_depth=depth, random_state=0))
    tree_model.fit(X, y)

    # Get predicted probabilities for each label
    tree_probs_train = tree_model.predict_proba(X)
    score_vec = np.column_stack([
        p[:, 1] if p.shape[1] > 1 else np.zeros(p.shape[0], dtype=bool)
         for p in tree_probs_train
    ]).astype(int)

    return tree_model, score_vec