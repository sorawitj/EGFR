import numpy as np
import pandas as pd
import datamol as dm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


def get_data(dir_path, active_threshold=8):
    """
    Load and prepare data from a CSV file.

    :param dir_path: Path to CSV file containing the dataset.
    :param active_threshold: Threshold to determine if a compound is active.
    :return: DataFrame with relevant columns ['molecule_chembl_id', 'smiles', 'is_active'].
    """
    # Load the dataset from a CSV file
    df = pd.read_csv(dir_path)

    # Create a new column 'is_active' to indicate if the pIC50 value is above the threshold (active or not)
    df['is_active'] = df['pIC50'] > active_threshold

    # Select relevant columns for further processing
    cols = ['molecule_chembl_id', 'smiles', 'is_active']

    return df[cols]


def _preprocess(row):
    """
    Preprocess a single molecule (SMILES representation).

    :param row: SMILES string of the molecule.
    :return: Preprocessed and standardized SMILES string.
    """
    # Convert the SMILES string to a molecular object
    mol = dm.to_mol(row, ordered=True)

    # Fix and sanitize the molecule
    mol = dm.fix_mol(mol)
    mol = dm.sanitize_mol(mol)

    # Standardize the molecule
    mol = dm.standardize_mol(mol)

    # Convert the molecule back to a standardized SMILES string
    return dm.standardize_smiles(dm.to_smiles(mol))


def prepare_data(df, transf):
    """
    Prepare the data for modeling (preprocessing and transformation).

    :param df: DataFrame with SMILES and "is_active" labels.
    :param transf: Transformer to convert SMILES into feature vectors.
    :return: Tuple (X, y) of feature vectors (X) and labels (y).
    """
    # Extract SMILES and activity labels from the DataFrame
    X, y = df["smiles"], df["is_active"]

    # Suppress RDKit warnings during molecule transformation
    with dm.without_rdkit_log():
        # Apply the _preprocess function to each SMILES string
        X = X.apply(_preprocess)

        # Convert the preprocessed SMILES into molecular features using a batch transformation
        X = np.stack(transf.batch_transform(transf, mols=X, batch_size=256))

    return X, y


def get_model_performance(model, X, y, featurizer_name, ax, config):
    """
    Evaluate model performance using cross-validation and plot ROC curves.

    :param model: Dictionary containing classifier ('clf') and hyperparameters ('h_params').
    :param X: Feature vectors.
    :param y: Labels.
    :param featurizer_name: Name of the featurizer used for the model.
    :param ax: Matplotlib axis object for plotting the ROC curve.
    :param config: Configuration dictionary specifying the number of folds.
    :return: Mean AUC score of the model.
    """
    # Get the number of folds for validation and testing from the config
    n_fold_val, n_fold_test = config['n_folds']['val'], config['n_folds']['test']

    # Create a StratifiedKFold object for test splits
    skf = StratifiedKFold(n_splits=n_fold_test)
    skf.get_n_splits(X, y)

    # Get the classifier and hyperparameters for tuning
    clf = model['clf']
    param_grid = model['h_params']

    # Define grid search with cross-validation for hyperparameter tuning
    cross_validation = StratifiedKFold(n_splits=n_fold_val, shuffle=True, random_state=0)
    grid_search = GridSearchCV(clf, param_grid=param_grid, scoring='roc_auc', cv=cross_validation, n_jobs=-1)

    # Initialize variables to store true positive rates and base false positive rates
    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    # Perform k-fold cross-validation for the test set
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        # Fit the model using the training data
        trained_clf = grid_search.fit(X[train_index], y[train_index])

        # Get the predicted probabilities for the test set
        y_score = trained_clf.predict_proba(X[test_index])

        # Compute the false positive rate (FPR) and true positive rate (TPR)
        fpr, tpr, _ = roc_curve(y[test_index], y_score[:, 1])

        # Plot individual ROC curve
        ax.plot(fpr, tpr, 'b', alpha=0.15)

        # Interpolate the TPRs at the base FPR values
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0  # Ensure the curve starts at (0,0)
        tprs.append(tpr)

    # Convert the list of TPRs into a numpy array and calculate mean and standard deviation
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    # Calculate upper and lower bounds for the ROC curve shading
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    # Compute the mean TPR and the area under the ROC curve (AUC)
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(base_fpr, mean_tpr)

    # Plot the mean ROC curve with AUC
    ax.plot(
        base_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f)" % mean_auc,
        lw=2,
        alpha=0.8,
    )

    # Shade the area between the upper and lower ROC curves
    ax.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    # Plot the random classifier line (diagonal)
    ax.plot([0, 1], [0, 1], 'r--')

    # Set plot limits and labels
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability",
    )

    # Add a legend and set the title to include model and feature information
    ax.legend()
    ax.set_title(f"Model: {model['name']}, featureizer: {featurizer_name}")

    return mean_auc
