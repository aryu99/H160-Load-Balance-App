# extract_data.py
# Usage: python extract_data.py /path/to/RGR_Load_sheet_160_1027_1.xlsm
import sys
import json
from pathlib import Path
import pandas as pd

def main(xlsx_path):
    xl = pd.ExcelFile(xlsx_path)
    df_data = xl.parse("Data", header=None)
    df_h = xl.parse("H160 L&T", header=None)

    def col_values(df, start_row, col, end_row=None):
        if end_row is None:
            end_row = start_row
        series = df.iloc[start_row-1:end_row, col-1]
        vals = [float(x) if pd.notna(x) else None for x in series.tolist()]
        return vals

    data_bundle = {}
    data_bundle["long_envelope"] = {"x": col_values(df_data, 6, 3, 12),
                                    "y": col_values(df_data, 6, 4, 12)}
    data_bundle["lat_envelope"] = {"x": col_values(df_data, 6, 6, 12),
                                   "y": col_values(df_data, 6, 7, 12)}
    data_bundle["limit_line"] = {"x": col_values(df_data, 21, 5, 22),
                                 "y": col_values(df_data, 21, 6, 22)}
    data_bundle["cg_long_points"] = {"x": col_values(df_data, 15, 3, 17),
                                     "y": col_values(df_data, 15, 4, 17)}
    data_bundle["cg_lat_points"] = {"x": col_values(df_data, 15, 6, 17),
                                    "y": col_values(df_data, 15, 7, 17)}

    fuel_block = df_data.iloc[3-1:134, 9-1:11].dropna(how="all").reset_index(drop=True)
    fuel_block.columns = ["Fuel_kg", "Arm_m", "Moment"]
    fuel_block["Fuel_kg"] = pd.to_numeric(fuel_block["Fuel_kg"], errors="coerce")
    fuel_block["Arm_m"] = pd.to_numeric(fuel_block["Arm_m"], errors="coerce")
    fuel_block["Moment"] = pd.to_numeric(fuel_block["Moment"], errors="coerce")
    data_bundle["fuel_table"] = fuel_block.to_dict(orient="records")

    # parse the H160 L&T table (items with arms)
    # find header row with "Item" and "Weight", then read down until blank item
    header_row_idx = None
    for i in range(0, 200):
        row = df_h.iloc[i].astype(str).str.lower().tolist()
        if any("item" in cell for cell in row) and any("weight" in cell for cell in row):
            header_row_idx = i
            break
    items = []
    if header_row_idx is not None:
        item_col = [j for j,val in enumerate(df_h.iloc[header_row_idx]) if isinstance(val, str) and 'item' in val.lower()][0]
        weight_col = [j for j,val in enumerate(df_h.iloc[header_row_idx]) if isinstance(val, str) and 'weight' in val.lower()][0]
        # find arm long & arm lat columns by header content
        arm_long_col = [j for j,val in enumerate(df_h.iloc[header_row_idx]) if isinstance(val, str) and 'arm long' in val.lower()]
        arm_lat_col = [j for j,val in enumerate(df_h.iloc[header_row_idx]) if isinstance(val, str) and 'arm lat' in val.lower()]
        arm_long_col = arm_long_col[0] if arm_long_col else weight_col+1
        arm_lat_col = arm_lat_col[0] if arm_lat_col else arm_long_col+2

        # data rows start two rows after header (there can be a blank)
        start = header_row_idx + 2
        for i in range(start, start+200):
            item = df_h.iat[i, item_col] if i < len(df_h) else None
            if pd.isna(item):
                break
            w = df_h.iat[i, weight_col] if i < len(df_h) else None
            arm_l = df_h.iat[i, arm_long_col] if i < len(df_h) else None
            # moment column likely is arm_long_col + 1
            moment_long = df_h.iat[i, arm_long_col+1] if i < len(df_h) else None
            arm_lat = df_h.iat[i, arm_lat_col] if i < len(df_h) else None
            moment_lat = df_h.iat[i, arm_lat_col+1] if i < len(df_h) else None
            items.append({"excel_row": i+1, "item": item, "weight": w,
                          "arm_long": arm_l, "moment_long": moment_long,
                          "arm_lat": arm_lat, "moment_lat": moment_lat})

    out = Path("app_data")
    out.mkdir(exist_ok=True)
    with open(out/"data_bundle.json", "w") as fh:
        json.dump(data_bundle, fh, indent=2)
    pd.DataFrame(data_bundle["fuel_table"]).to_csv(out/"fuel_table.csv", index=False)
    with open(out/"h160_items.json", "w") as fh:
        json.dump(items, fh, indent=2)

    print("Saved app_data/data_bundle.json, app_data/h160_items.json, app_data/fuel_table.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_data.py /path/to/excel.xlsm")
    else:
        main(sys.argv[1])
