def rename_with_suffix(df, fields, suffix):
    rename_map = {field: f"{field}_{suffix}" for field in fields}
    return df.rename(columns=rename_map)
