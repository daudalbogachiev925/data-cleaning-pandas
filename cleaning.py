"""Полный пайплайн очистки данных."""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_data(df):
    logger.info(f"Исходный размер: {df.shape}")
    logger.info(f"Пропуски:\n{df.isnull().sum()}")

    # 1. Удаление дубликатов
    df = df.drop_duplicates()
    logger.info(f"После удаления дубликатов: {df.shape}")

    # 2. Обработка пропусков
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("unknown")

    # 3. Удаление выбросов (IQR)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    logger.info(f"После очистки: {df.shape}")
    return df

if __name__ == "__main__":
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(18, 80, 1000).astype(float),
        "salary": np.random.normal(50000, 15000, 1000),
        "city": np.random.choice(["Moscow", "SPb", None], 1000),
    })
    df.loc[::50, "age"] = np.nan
    df = pd.concat([df, df.iloc[:10]])

    cleaned = clean_data(df)
    cleaned.to_csv("cleaned.csv", index=False)
