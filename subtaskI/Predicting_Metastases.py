import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
import torch
from torch._prims_common import DeviceLikeType
from BoostedMetastasisNN import train_nn
from Decision_tree import decision_tree_per_label
from src.preprocess import preprocess, prepare_features

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
from ImputerPipeline import ImputerPipeline
from fix_train_labels import convert_binary_to_labels_csv, convert_labels_to_binary_csv
from pandas import DataFrame

import json
import joblib
import torch.onnx

from itertools import product

# constants
SEED = 100
LABELS = [
    "ADR - Adrenals",
    "BON - Bones",
    "BRA - Brain",
    "HEP - Hepatic",
    "LYM - Lymph nodes",
    "MAR - Bone Marrow",
    "OTH - Other",
    "PER - Peritoneum",
    "PLE - Pleura",
    "PUL - Pulmonary",
    "SKI - Skin"
]
HYPER_PARAMS_TRAIN = {
    "tree_depth": [3, 5],
    "hidden_dims": [(64,), (64, 32)],
    "n_epochs": [10, 20],
    "lr": [1e-3, 1e-4]
}


def cross_validate_pipeline(X, y, param_grid, original_X: DataFrame, device: DeviceLikeType, cv=3):
    best_score = -np.inf
    best_params = None

    for depth in param_grid["tree_depth"]:
        
        for hidden_dims in param_grid["hidden_dims"]:
            for n_epochs in param_grid["n_epochs"]:
                for lr in param_grid["lr"]:
                    scores = []

                    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
                    for train_idx, val_idx in kf.split(X):
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]

                        # ----- impute/encode only on training fold ----------
                        processed_X_train = pd.DataFrame(X_train, columns=original_X.columns)
                        processed_X_val = pd.DataFrame(X_val, columns=original_X.columns)
                        processed_X_train = prepare_features(processed_X_train)
                        processed_X_val = prepare_features(processed_X_val)
                        pre = ImputerPipeline(processed_X_train)
                        X_train = pre.fit_transform(processed_X_train)
                        X_val = pre.transform(processed_X_val)
                        
                        _, tree_train = decision_tree_per_label(X_train, y_train, depth=depth)
                        _, tree_val = decision_tree_per_label(X_val, y_val, depth=depth)

                        print(f"Finished decision tree for depth={depth}, "
                              f"hidden={hidden_dims}, epochs={n_epochs}, lr={lr}")

                        X_boosted_train = np.hstack([X_train, tree_train])
                        X_boosted_val = np.hstack([X_val, tree_val])

                        model = train_nn(
                            X_boosted_train, y_train, device,
                            hidden_dims=hidden_dims,
                            n_epochs=n_epochs, lr=lr, batch_size=128
                        )
                        model.eval()

                        print(f"Finished training NN for depth={depth}, "
                              f"hidden={hidden_dims}, epochs={n_epochs}, lr={lr}")

                        preds = model(torch.tensor(X_boosted_val, dtype=torch.float32, device=device))\
                            .cpu().detach().numpy()
                        preds_binary = (preds > 0.5).astype(int)

                        # Calculate loss (binary cross-entropy)
                        import torch.nn as nn
                        criterion = nn.BCELoss()
                        val_tensor = torch.tensor(preds, dtype=torch.float32, device=device)
                        y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
                        loss = criterion(val_tensor, y_val_tensor).item()

                        score = f1_score(y_val, preds_binary, average="macro")
                        print(f"Fold F1 Score: {score:.3f}, Fold BCE Loss: {loss:.3f} for depth={depth}, "
                                f"hidden={hidden_dims}, epochs={n_epochs}, lr={lr}")
                        scores.append(score)

                        # Calculate current combination index for progress
                        # Calculate progress based on total number of parameter combinations and folds
                        # Compute the total number of combinations (cartesian product)
                        param_combos = list(product(
                            param_grid["tree_depth"],
                            param_grid["hidden_dims"],
                            param_grid["n_epochs"],
                            param_grid["lr"]
                        ))
                        total_combinations = len(param_combos) * cv

                        # Find the current combination index
                        combo_idx = param_combos.index((depth, hidden_dims, n_epochs, lr))
                        # Fold index is kf.split(X) order, so use len(scores) as fold number (starts at 1)
                        fold_idx = len(scores)
                        current_idx = combo_idx * cv + fold_idx
                        percent = 100 * current_idx / total_combinations
                        print(f"Progress: {percent:.1f}% ({current_idx}/{total_combinations})")

                    avg_score = np.mean(scores)
                    print(f"Params: depth={depth}, hidden={hidden_dims}, epochs={n_epochs}, lr={lr} → F1={avg_score:.3f}")
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {
                            "tree_depth": depth,
                            "hidden_dims": hidden_dims,
                            "n_epochs": n_epochs,
                            "lr": lr
                        }

    return best_params


def tree_to_dict(tree):
    """
    Recursively traverses a scikit-learn decision tree and converts it to a
    JSON-serializable dictionary. Util to be used for exporting the tree.
    """
    tree_ = tree.tree_

    def recurse(node_id):
        # A node is a leaf if its left and right children are the same
        if tree_.children_left[node_id] == tree_.children_right[node_id]:
            # The 'value' attribute for a classifier tree node contains the
            # number of training samples that fall into each class.
            # e.g., [[10., 50.]] means 10 samples for class 0, 50 for class 1.
            return {"value": tree_.value[node_id][0].tolist()}

        # A node is a split node
        else:
            feature_index = tree_.feature[node_id]
            threshold = tree_.threshold[node_id]
            left_child_id = tree_.children_left[node_id]
            right_child_id = tree_.children_right[node_id]
            return {
                "feature_index": int(feature_index),
                "threshold": float(threshold),
                "left": recurse(left_child_id),
                "right": recurse(right_child_id)
            }
    # Start the recursion from the root node (ID 0)
    return recurse(0)

def save_model(imputer, tree_model, nn_model, X_data, device='gpu', output_prefix=""):
    """
    Saves the model to a file using joblib.
    """
    # ----- Save the preprocessor and tree model -----
     # Save the fitted preprocessor pipeline
    joblib.dump(imputer, output_prefix + 'imputer_pipeline.joblib')
    print(f"✅ Preprocessor pipeline saved to {output_prefix}imputer_pipeline.joblib")

    # Save the original sklearn tree model
    joblib.dump(tree_model, output_prefix + 'tree_model.joblib')
    print(f"✅ Decision Tree model saved to {output_prefix}tree_model.joblib")
    
    # Export the tree structure to a JSON file for the web app
    # We loop through each tree (one for each label) and use our helper function
    tree_json_list = [tree_to_dict(estimator) for estimator in tree_model.estimators_]
    with open(output_prefix + 'tree_model.json', 'w') as f:
        json.dump(tree_json_list, f)
    print(f"✅ Decision Tree structure exported to {output_prefix}tree_model.json for web use.")

    nn_model.eval() # Set the model to evaluation mode
    
    # Create a dummy input with the correct shape for the export
    # The shape is (batch_size, num_features + num_tree_outputs)
    dummy_input_shape = (1, X_data.shape[1]) 
    dummy_input = torch.randn(dummy_input_shape, device=device)

    # Export the model to ONNX format
    torch.onnx.export(nn_model,
                      dummy_input,
                      output_prefix + "boosted_nn.onnx",
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})
    print(f"✅ Neural Network model exported to {output_prefix}boosted_nn.onnx")

def train(X_input, y_input, predictions_output="train_predictions.csv", y_output="", sample=0):
    # Load training and test data
    X = pd.read_csv(X_input)
    y = convert_labels_to_binary_csv(input_path=y_input)
    X = prepare_features(X)

    assert len(X) == len(y), f"X({len(X)}) and y({len(y)}) must have the same number of samples"

    if sample > 0:
        # Sample sample% of the data
        X_sampled = X.sample(frac=sample, random_state=SEED)
        y_sampled = y.loc[X_sampled.index]
        X_train, X_test, y_train, y_test = train_test_split(
            X_sampled.values, y_sampled.values, test_size=0.2, random_state=SEED
        )
    else:
        # Use the full dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=0.2, random_state=SEED
        )

    print("cuda available: ", torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----- Step 0: Hyperparameter tuning -----
    best_params = cross_validate_pipeline(X_train, y_train, HYPER_PARAMS_TRAIN, X, device)
    print("✅ Best hyperparameters:", best_params)
    
    # ----- Step 1: Impute/encode -----
    processed_X_train = pd.DataFrame(X_train, columns=X.columns)
    processed_X_test = pd.DataFrame(X_test, columns=X.columns)
    processed_X_train = prepare_features(processed_X_train)
    processed_X_test = prepare_features(processed_X_test)
    pre = ImputerPipeline(processed_X_train)
    X_train = pre.fit_transform(processed_X_train)
    X_test = pre.transform(processed_X_test)

    # ----- Step 2: Final training -----
    tree_model, tree_train = decision_tree_per_label(X_train, y_train, depth=best_params["tree_depth"])

    X_boosted = np.hstack([X_train, tree_train])
    model = train_nn(
        X_boosted, y_train, device,
        hidden_dims=best_params["hidden_dims"],
        n_epochs=best_params["n_epochs"],
        lr=best_params["lr"]
    )
    print("✅ Model trained with best hyperparameters")

    # Save the model and preprocessor
    save_model(pre, tree_model, model, X_boosted,
               device=device, output_prefix=str(sample) if sample > 0 else "")

    # ----- Step 3: Predict on test -----
    tree_probs_test = tree_model.predict_proba(X_test)
    tree_test = np.column_stack([
        p[:, 1] if p.shape[1] > 1 else np.zeros(p.shape[0], dtype=bool)
         for p in tree_probs_test
    ]).astype(int)
    X_test_boosted = np.hstack([X_test, tree_test])

    model.eval()
    test_tensor = torch.tensor(X_test_boosted, dtype=torch.float32,
                               device=device)
    with torch.no_grad():
        test_probs = model(test_tensor).cpu().numpy()
    test_preds = (test_probs > 0.5).astype(int)
    print("✅ Predictions made on test set")

    predictions = pd.DataFrame(test_preds)
    predictions = convert_binary_to_labels_csv(predictions, output_path=predictions_output)
    if (y_output):
        convert_binary_to_labels_csv(pd.DataFrame(y_test), output_path=y_output)
    print("🎉 predictions saved!")

def predict(test_input="test.feats.csv", processed_X_data="cleaned_train.feats.csv",
            output_path="test_predictions.csv"):
    X = pd.read_csv(processed_X_data)
    test = pd.read_csv(test_input)
    preprocessed_test = preprocess(test)
    processed_test = prepare_features(preprocessed_test)

    # 1) Reload preprocessor and tree model
    pre: ImputerPipeline = joblib.load("imputer_pipeline.joblib")
    tree_model:MultiOutputClassifier = joblib.load("tree_model.joblib")

    X_test = pd.DataFrame(processed_test, columns=X.columns)
    # impute/encode:
    processed_X_test = pre.transform(X_test)

    # 3) Generate the tree boost features
    tree_probs = tree_model.predict_proba(processed_X_test)
    tree_feats = np.column_stack([
        p[:, 1] if p.shape[1] > 1 else np.zeros(p.shape[0], dtype=bool)
         for p in tree_probs
    ]).astype(int)

    # 4) Stack original + tree bits
    boosted_X = np.hstack([processed_X_test, tree_feats])

    # 5) Load NN model and run against tree-boosted features
    import onnxruntime as ort
    sess = ort.InferenceSession("boosted_nn.onnx")
    inp  = sess.get_inputs()[0].name
    outp = sess.get_outputs()[0].name
    probs = sess.run([outp], {inp: boosted_X.astype(np.float32)})[0]
    
    print("✅ Predictions calculated")

    # 6) Threshold to binary and map back to labels
    preds_binary = (probs > 0.5).astype(int)
    predictions = pd.DataFrame(preds_binary, columns=LABELS)
    convert_binary_to_labels_csv(predictions, output_path=output_path)
    print("🎉 test_predictions.csv saved!")



if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Predict breast-cancer spread locations")
    ap.add_argument("method", help="Choose whether to [T]rain or [P]redict")
    ap.add_argument("x", help="Path to X (features) input")
    ap.add_argument("--y", help="Path to y (labels) input")
    ap.add_argument("--test", help="Path to test (features) input")
    ap.add_argument("--pred_out", help="Path to output predictions")
    ap.add_argument("--y_out", help="Path to output (processed) y data (optional)")
    ap.add_argument("--sample", help="Optional sample size")
    args = ap.parse_args()

    print(args)
    method = args.method
    if method == "T":
        train(
            X_input=args.x,
            y_input=args.y,
            predictions_output=args.pred_out,
            y_output=args.y_out if args.y_out else "",
            sample=float(args.sample) if args.sample else 0
        )
    elif method == "P":
        predict(
            test_input=args.test,
            processed_X_data=args.x,
            output_path=args.pred_out
        )
    else:
        print("Invalid method. Use 'T' for train or 'P' for predict.")