from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Flexible preprocessor (imputation + encoding + scaling)
def get_preprocessor(scaler_type='MinMax', numerical_cols=None, categorical_cols=None):
    if scaler_type == 'MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute numerical (missing: BuildingArea, etc.)
        ('scaler', scaler)  # Scaling
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute categorical (missing: CouncilArea)
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Encoding
    ])

    return ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])