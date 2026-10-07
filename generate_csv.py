import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def preprocess_classification(dataset: str, df: pd.DataFrame, diseases: tuple) -> pd.DataFrame:
    """
    Given a dataset name and a dataframe, filter & rename columns for classification tasks; prepare for train/val/test split

    Args:
        dataset: str - the dataset name (e.g. chestxray-8)
        df: pd.DataFrame - the dataframe to preprocess

    Returns:
        pd.DataFrame - the preprocessed dataframe
    """

    if dataset.lower() == 'chestxray-8':
        mapping = {diseases[0]: '0', diseases[1]: '1'}
        paths, labels = 'Image Index', 'Finding Labels'
        mask = (
            df[labels]
            .str.split("|")  # Split multi-label entries
            .apply(lambda x: len(x) == 1 and x[0] in mapping.keys())
        )
        filtered_df = df[mask]  # Keep only valid rows
        
        filtered_cols = filter_cols(filtered_df, mapping, cols=(paths, labels))
        filtered_cols = filtered_cols.rename(columns={paths: 'image_path', labels: 'label'})
        return filtered_cols

def preprocess_segmentation(dataset: str, df: pd.DataFrame):
    if dataset.lower() == 'shenzhen':
        pass

    elif dataset.lower() == 'landmark':
        pass

    elif dataset.lower() == 'montgomery':
        pass

# All
def split(filtered_cols: pd.DataFrame) -> pd.DataFrame: 
    """
    Given a filtered dataframe, split the data into train, validation, and test sets
    Enforces a 70% train, 15% validation, 15% test split

    Args:
        filtered_cols: pd.DataFrame - the filtered dataframe to partition into train, validation, and test sets
    
    Returns:
        pd.DataFrame - the final dataframe with the split column added

    """
    train, temp = train_test_split(filtered_cols, test_size=0.3, random_state=42, stratify=filtered_cols['label'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])
    train['split'], val['split'], test['split'] = 'train', 'val', 'test'
    final_df = pd.concat([train, val, test])
    return final_df

# Classification (Chestxray-8)
def filter_cols(df: pd.DataFrame, mapping: dict, cols = ('Image Index', 'Finding Labels')) -> pd.DataFrame:
    """
    Given a dataframe, filter the columns based on the original author's column names & map diseases to binary values

    Args:
        df: pd.DataFrame - the dataframe to filter
        mapping: dict - the mapping of diseases to binary values
        cols: tuple - the columns to filter by (assumes the first column is the image path & the second is the labels)

    Returns:
        pd.DataFrame - the filtered dataframe
    """
    labels = cols[1]
    columns = df[list(cols)]
    filtered_cols = columns[columns[labels].isin(mapping.keys())].copy()
    filtered_cols.loc[:, 'disease'] = filtered_cols[labels]
    filtered_cols.loc[:, labels] = filtered_cols[labels].map(mapping)
    return filtered_cols

# Classification (Chestxray-8)
def chex_image_path(base_path) -> set: 
    """
    Given the base path, return the set of full image paths based on the following directory structure:
    ..base_path/images0i/images/image_name for i in range(1, 13)
    (currently for Chestxray-8 specifically)
    
    Args:
        base_path: str - the base path to the dataset (i.e. chestxray-8)
    
    Returns:
        set - the set of full image paths
    """
    image_paths = set()
    for i in range(1, 13):
        folder = os.path.join(base_path, f'images{i:02d}')
        for root, _, files in os.walk(folder):
            for file in files:
                image_paths.add(os.path.join(root, file))
    return image_paths

def get_full_image_path(image_name: str,image_paths: set) -> str:
    """
    Given an image name, return the full path to the image
    Sample in: get_full_image_path('00000001_000.png', image_paths)
    Sample out: '../chestxray-8/images01/00000001_000.png'

    Args:
        image_name: str - the name of the image
        image_paths: set - the set of full image paths
    Returns:
        str - the full path to the image if it exists in the set, otherwise the image name
    """
    for path in image_paths:
        if image_name in path:
            return path
    raise ValueError(f"{image_name} not found in image paths")

if __name__ == '__main__':
    # TODO ~ Customize
    parser = argparse.ArgumentParser(description="Generate training CSVs for datasets")
    parser.add_argument("--task", type=str, default="classification", help="Classification or segmentation (classification- no need for other args)")
    parser.add_argument("--dataset", type=str, default="chestxray-8", help="Dataset name")
    parser.add_argument("--diseases", type=str, default="Effusion,Nodule", help="Diseases to generate dataset for")

    # Sample file structure with params task='classification', dataset='chestxray-8'
    # input: ../chestxray-8/chestxray-8.csv
    # output: ../chestxray-8/chestxray-8-labels.csv

    args = parser.parse_args()
    dataset = args.dataset
    csv = f'{dataset}.csv'.lower()
    diseases = tuple(args.diseases.split(","))
    disease_0 = diseases[0].lower()[:3]
    disease_1 = diseases[1].lower()[:3]
    output_name = f"{dataset}-{disease_0}-{disease_1}-labels.csv".lower()
    base_path = f'../{dataset}'

    data = pd.read_csv(f"../{dataset}/{csv}")
    df = preprocess_classification(dataset, data, diseases)
    final_df = split(df)
    image_paths = chex_image_path(base_path)
    final_df['image_path'] = final_df['image_path'].apply(get_full_image_path, args = (image_paths,))

    csv_file = f"../{dataset}/{output_name}"
    final_df.to_csv(csv_file, index=False)


