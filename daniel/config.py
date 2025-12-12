KAGGLE_DATASET_ID = "firecastrl/us-wildfire-dataset"
KAGGLE_FILE_NAME = "Wildfire_Dataset.csv"
PROCESSED_X_PATH = "data/processed/X_compressed.npy"
PROCESSED_Y_PATH = "data/processed/y.npy"

ID_COL = "sample_id"

TIME_COL = "datetime"

TARGET_COL = "Wildfire"

STATIC_COLS = ["latitude", "longitude"]

TIME_SERIES_FEATURE_COLS = [
    "pr",
    "rmax",
    "rmin",
    "sph",     
    "srad",    
    "tmmn",
    "tmmx",
    "vs",      
    "bi",      
    "fm100", 
    "fm1000",
    "erc",
    "etr",     
    "pet",     
    "vpd",
]

KMEANS_K = 3
RANDOM_STATE = 42
WINDOW_LENGTH = 75