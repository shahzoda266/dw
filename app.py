import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

st.set_page_config(page_title="Data Wrangler", layout="wide")
st.title("Data Wrangler & Visualizer")

# ================= NAVIGATION =================
page = st.sidebar.selectbox(
    "Navigation",
    ["Upload & Overview", "Cleaning & Preparation"]
)

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "json"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_json(uploaded_file)

    st.session_state.df = df.copy()

if "df" in st.session_state:
    df = st.session_state.df

if "log" not in st.session_state:
    st.session_state.log = []

# ================= PAGE A =================
if page == "Upload & Overview":

    if "df" not in st.session_state:
        st.warning("Upload dataset first")
    else:
        st.dataframe(df)

        st.write("Rows:", df.shape[0])
        st.info(f"Number of columns: {df.shape[1]}")

        st.markdown("---")
        st.subheader("Column Types")

        def simplify_dtype(dtype):
            if "int" in str(dtype) or "float" in str(dtype):
                return "Numeric"
            elif "datetime" in str(dtype):
                return "Date"
            else:
                return "Text"

        col_types = pd.DataFrame({
            "Column": df.columns,
            "Type": [simplify_dtype(dtype) for dtype in df.dtypes]
        })

        st.dataframe(col_types)

        st.markdown("---")
        st.subheader("Summary Statistics (Numeric)")

        num_df = df.select_dtypes(include=np.number)

        if not num_df.empty:
            st.dataframe(num_df.describe())
        else:
            st.info("No numeric columns available")


        st.markdown("---")
        st.subheader("Summary Statistics (Categorical)")

        cat_df = df.select_dtypes(include=["object", "category"])

        if not cat_df.empty:
            summary_cat = pd.DataFrame({
                "Column": cat_df.columns,
                "Unique Values": [cat_df[col].nunique() for col in cat_df.columns],
                "Most Frequent": [cat_df[col].mode()[0] if not cat_df[col].mode().empty else None for col in cat_df.columns]
            })
            st.dataframe(summary_cat)
        else:
            st.info("No categorical columns available")

        st.markdown("---")
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        percent = (missing / len(df)) * 100
        st.dataframe(pd.DataFrame({"Missing": missing, "%": percent}))

        st.subheader("Duplicates")
        st.write(df.duplicated().sum())

        if st.button("Reset Session"):
            st.session_state.clear()

# ================= PAGE B =================
elif page == "Cleaning & Preparation":

    if "df" not in st.session_state:
        st.warning("Upload data first")
    else:

        num_cols = df.select_dtypes(include=np.number).columns

        st.header("Cleaning Tools")

        # ===== TYPE CONVERSION =====
        st.subheader("Data Type Conversion")

        col = st.selectbox("Column", df.columns)
        dtype = st.selectbox("Convert to", ["Numeric","Datetime","Categorical"])

        fmt = st.text_input("Datetime format (optional)") if dtype=="Datetime" else None

        clean_option = []
        if dtype == "Numeric":
            clean_option = st.multiselect(
                "Clean numeric issues",
                ["Remove commas (,)", "Remove currency symbols ($, €, £)", "Remove spaces"]
            )

        if st.button("Apply Conversion"):
            if dtype == "Numeric":
                temp = df[col].astype(str)

                if "Remove commas (,)" in clean_option:
                    temp = temp.str.replace(",", "")
                if "Remove currency symbols ($, €, £)" in clean_option:
                    temp = temp.str.replace(r"[€$£]", "", regex=True)
                if "Remove spaces" in clean_option:
                    temp = temp.str.replace(" ", "")

                df[col] = pd.to_numeric(temp, errors="coerce")

            elif dtype == "Datetime":
                df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")

            elif dtype == "Categorical":
                df[col] = df[col].astype("category")

            st.session_state.df = df
            st.session_state.log.append(f"Converted {col} to {dtype}")
            st.success("Conversion applied")

        # ===== MISSING VALUES =====
        st.subheader("Missing Values Handling")

        missing = df.isnull().sum()
        percent = (missing / len(df)) * 100
        st.dataframe(pd.DataFrame({"Missing": missing, "%": percent}))

        col = st.selectbox("Select column to handle missing values", df.columns)

        action = st.selectbox(
            "Action",
            ["Drop Rows","Mean","Median","Mode","Most Frequent","Constant","Forward Fill","Backward Fill"]
        )

        val = st.text_input("Constant value") if action=="Constant" else None

        before_rows = len(df)
        before_missing = df[col].isnull().sum()

        if st.button("Apply Missing Handling"):

            if action in ["Mean","Median"] and col not in num_cols:
                st.error("Mean/Median can only be applied to numeric columns")
                st.stop()

            if action == "Drop Rows":
                df = df.dropna(subset=[col])
            elif action == "Mean":
                df[col] = df[col].fillna(df[col].mean())
            elif action == "Median":
                df[col] = df[col].fillna(df[col].median())
            elif action in ["Mode","Most Frequent"]:
                df[col] = df[col].fillna(df[col].mode()[0])
            elif action == "Constant":
                df[col] = df[col].fillna(val)
            elif action == "Forward Fill":
                df[col] = df[col].fillna(method="ffill")
            elif action == "Backward Fill":
                df[col] = df[col].fillna(method="bfill")

            after_rows = len(df)
            after_missing = df[col].isnull().sum()

            st.success("Changes applied")
            st.write("### Before vs After")
            st.write(f"Rows: {before_rows} → {after_rows}")
            st.write(f"Missing in '{col}': {before_missing} → {after_missing}")

            st.session_state.df = df
            st.session_state.log.append(f"Handled missing in {col} using {action}")

        # ===== DROP COLUMNS =====
        st.subheader("Drop Columns by Missing %")

        threshold = st.slider("Drop columns above % missing", 0, 100, 50)

        if st.button("Drop Columns"):
            before_cols = df.shape[1]
            to_drop = df.columns[df.isnull().mean()*100 > threshold]
            df = df.drop(columns=to_drop)
            after_cols = df.shape[1]

            st.write("### Before vs After")
            st.write(f"Columns: {before_cols} → {after_cols}")
            st.write("Dropped columns:", list(to_drop))

            st.session_state.df = df
            st.session_state.log.append(f"Dropped columns > {threshold}% missing")

        # ===== DUPLICATES =====
        st.subheader("Duplicates Handling")

        mode = st.radio("Duplicate type", ["Full Row","Subset"])

        if mode == "Subset":
            cols = st.multiselect("Select columns", df.columns)
            if not cols:
                st.warning("Please select at least one column.")
        else:
            cols = None

        if mode == "Full Row":
            dup = df.duplicated()
            dup_all = df[df.duplicated(keep=False)]
        elif mode == "Subset" and cols:
            dup = df.duplicated(subset=cols)
            dup_all = df[df.duplicated(subset=cols, keep=False)]
        else:
            dup = pd.Series([False]*len(df))
            dup_all = pd.DataFrame()

        st.write(f"Duplicates found: {dup.sum()}")

        if st.checkbox("Show duplicate rows"):
            if not dup_all.empty:
                dup_all = dup_all.copy()
                if mode == "Full Row":
                    dup_all["Duplicate_Group"] = dup_all.groupby(list(df.columns)).ngroup()
                else:
                    dup_all["Duplicate_Group"] = dup_all.groupby(cols).ngroup()
                st.dataframe(dup_all.sort_values("Duplicate_Group"))
            else:
                st.write("No duplicates found.")

        keep = st.selectbox("Keep", ["First","Last"])

        if st.button("Remove Duplicates"):
            before = len(df)
            if mode == "Full Row":
                df = df.drop_duplicates(keep="first" if keep=="First" else "last")
            elif mode == "Subset" and cols:
                df = df.drop_duplicates(subset=cols, keep="first" if keep=="First" else "last")

            after = len(df)
            st.write(f"Rows: {before} → {after}")

            st.session_state.df = df
            st.session_state.log.append(f"Removed duplicates ({mode}, keep {keep})")
            st.success("Duplicates removed")

        # ===== CATEGORICAL TOOLS =====
        st.subheader("Categorical Tools")

        cat_cols = df.select_dtypes(include=["object","category"]).columns

        if len(cat_cols) > 0:

            c = st.selectbox("Categorical column", cat_cols)

            std_action = st.selectbox("Standardize", ["None","Lower","Upper","Title","Trim"])

            if st.button("Apply Standardization"):

                if std_action == "None":
                    st.warning("Please select a standardization option.")
                else:
                    if std_action == "Lower":
                        df[c] = df[c].str.lower()
                    elif std_action == "Upper":
                        df[c] = df[c].str.upper()
                    elif std_action == "Title":
                        df[c] = df[c].str.title()
                    elif std_action == "Trim":
                        df[c] = df[c].str.strip()

                    st.session_state.df = df
                    st.session_state.log.append(f"Standardized {c}")

                    st.success("Standardization applied")
                    st.dataframe(df[[c]].head(10))

            mapping_text = st.text_area("Mapping old:new")

            if st.button("Apply Mapping"):
                mapping_dict = {}

                for line in mapping_text.split("\n"):
                    if ":" in line:
                        k, v = line.split(":")
                        mapping_dict[k.strip()] = v.strip()

                if not mapping_dict:
                    st.warning("Please enter valid mappings in format old:new")
                else:
                    df[c] = df[c].map(mapping_dict).fillna(df[c])

                    st.session_state.df = df
                    st.session_state.log.append(f"Mapped values in {c}")

                    st.success("Mapping applied")
                    st.dataframe(df[[c]].head(10))

            thresh = st.slider("Rare category threshold %", 0, 20, 5)

# Preview BEFORE applying
            freq = df[c].value_counts(normalize=True) * 100
            rare = freq[freq < thresh].index
            
            selected = st.multiselect(
                "Select categories to group into 'Other'",
                options=list(rare)
            )

            if st.button("Group Rare"):

                if len(selected) == 0:
                    st.info("No categories selected")

                else:
                    before_counts = df[c].value_counts()

                    df[c] = df[c].replace(selected, "Other")

                    after_counts = df[c].value_counts()

                    st.session_state.df = df
                    st.session_state.log.append(f"Grouped rare in {c} (<{thresh}%)")
  
                    st.success("Rare categories grouped")

                    st.write("### Before")
                    st.dataframe(before_counts.head(10))

                    st.write("### After")
                    st.dataframe(after_counts.head(10))

            if st.button("One-hot Encode"):
                df = pd.get_dummies(df, columns=[c])

                st.session_state.df = df
                st.session_state.log.append(f"Encoded {c}")

                st.success("One-hot encoding applied")
                st.dataframe(df.head(10))

        # ===== OUTLIERS =====
        st.subheader("Outlier Detection & Handling")

        if len(num_cols) > 0:

            c = st.selectbox("Select numeric column", num_cols, key="outlier_col")

            Q1 = df[c].quantile(0.25)
            Q3 = df[c].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = df[(df[c] < lower) | (df[c] > upper)]

            action = st.selectbox(
                "Choose action",
                ["Do Nothing", "Remove Rows", "Cap (Winsorize)"],
                key="outlier_action"
            )

            if st.button("Apply Outlier Handling"):

                before_rows = len(df)

                if action == "Do Nothing":
                    st.info("No changes applied")

                elif len(outliers) == 0:
                    st.info("No outliers detected — no changes needed")
                else:
                    if action == "Remove Rows":
                        df = df[(df[c] >= lower) & (df[c] <= upper)]

                    elif action == "Cap (Winsorize)":
                         df[c] = np.where(df[c] < lower, lower,
                                          np.where(df[c] > upper, upper, df[c]))

                    after_rows = len(df)

                    st.session_state.df = df
                    st.session_state.log.append(f"Outlier handling on {c}: {action}")

                    st.success("Outlier handling applied")
                    st.write("### Impact")
                    st.write(f"Outliers detected: {len(outliers)}")
                    st.write(f"Rows: {before_rows} → {after_rows}")
        else:
            st.info("No numeric columns available")

        # ===== SCALING =====
        st.subheader("Scaling")

        if len(num_cols) > 0:
            c = st.selectbox("Select numeric column to scale", num_cols, key="scale_col")
            m = st.selectbox("Method", ["MinMax","Z-score"], key="scale_method")

            before_mean = df[c].mean()
            before_std = df[c].std()

            if st.button("Apply Scaling"):

                scaler = MinMaxScaler() if m == "MinMax" else StandardScaler()
                df[[c]] = scaler.fit_transform(df[[c]])

                after_mean = df[c].mean()
                after_std = df[c].std()

                st.session_state.df = df
                st.session_state.log.append(f"Scaled {c} using {m}")

                st.success("Scaling applied")

                st.write("### Before vs After")
                st.write(f"Mean: {before_mean:.2f} → {after_mean:.2f}")
                st.write(f"Std Dev: {before_std:.2f} → {after_std:.2f}")

        else:
            st.info("No numeric columns available")


        # ===== COLUMN OPERATIONS =====
        st.subheader("Column Operations")

        # ---- Rename Column ----
        st.write("### Rename Column")

        old_name = st.selectbox("Select column to rename", df.columns, key="rename_col")
        new_name = st.text_input("New column name", key="rename_input")

        if st.button("Rename Column"):
            if new_name:
                df = df.rename(columns={old_name: new_name})

                st.session_state.df = df
                st.session_state.log.append(f"Renamed {old_name} to {new_name}")

                st.success("Column renamed")
            else:
                st.warning("Enter a new name")


        # ---- Create New Column ----
        st.write("### Create New Column")

        col1 = st.selectbox("Column A", df.columns, key="colA")
        col2 = st.selectbox("Column B", df.columns, key="colB")

        operation = st.selectbox("Operation", ["Add","Subtract","Multiply","Divide"], key="operation")

        new_col = st.text_input("New column name", key="create_col_input")

        if st.button("Create Column"):
            if new_col:
                if operation == "Add":
                    df[new_col] = df[col1] + df[col2]
                elif operation == "Subtract":
                    df[new_col] = df[col1] - df[col2]
                elif operation == "Multiply":
                    df[new_col] = df[col1] * df[col2]
                elif operation == "Divide":
                    df[new_col] = df[col1] / df[col2]

                st.session_state.df = df
                st.session_state.log.append(f"Created {new_col}")

                st.success("New column created")
                st.dataframe(df[[new_col]].head())
            else:
                st.warning("Enter column name")


                # ---- Binning ----
        st.write("### Binning")

        if len(num_cols) > 0:
            col_bin = st.selectbox("Select numeric column for binning", num_cols, key="bin_col")
            bins = st.slider("Number of bins", 2, 10, 4, key="bin_slider")

            method = st.selectbox("Method", ["Equal Width","Quantile"], key="bin_method")

            bin_col_name = st.text_input("New binned column name", key="bin_input")

            if st.button("Apply Binning"):
                if not bin_col_name:
                    st.warning("Enter column name")
                else:
                    try:
                        # Create bins first
                        if method == "Equal Width":
                            binned = pd.cut(df[col_bin], bins=bins)
                        else:
                            binned = pd.qcut(df[col_bin], q=bins, duplicates="drop")

                        # Create clean labels
                        labels = []
                        for interval in binned.cat.categories:
                            left = round(interval.left, 2)
                            right = round(interval.right, 2)
                            labels.append(f"{left} - {right}")

                        # Apply bins with labels
                        if method == "Equal Width":
                            df[bin_col_name] = pd.cut(df[col_bin], bins=bins, labels=labels)
                        else:
                            df[bin_col_name] = pd.qcut(df[col_bin], q=bins, labels=labels, duplicates="drop")

                        st.session_state.df = df
                        st.session_state.log.append(f"Binned {col_bin} into {bins} bins ({method})")

                        st.success("Binning applied")

                        st.write("### Preview")
                        st.dataframe(df[[col_bin, bin_col_name]].head(10))

                        st.write("### Bin Counts")
                        st.dataframe(df[bin_col_name].value_counts())

                    except Exception as e:
                        st.error(f"Binning failed: {e}")

        else:
            st.info("No numeric columns available")
                # ===== DATA VALIDATION RULES =====
                # ===== DATA VALIDATION RULES =====
        st.markdown("---")
        st.subheader("Data Validation Rules")

        validation_type = st.selectbox(
            "Select validation type",
            ["Numeric Range", "Allowed Categories", "Non-null Constraint"],
            key="val_type"
        )

        col_val = st.selectbox(
            "Select column",
            df.columns,
            key="val_col"
        )

        violations = pd.DataFrame()

        # Inputs depending on type
        if validation_type == "Numeric Range":
            min_val = st.number_input(
                "Minimum value",
                value=float(df[col_val].min()),
                key="val_min"
            )
            max_val = st.number_input(
                "Maximum value",
                value=float(df[col_val].max()),
                key="val_max"
            )
        elif validation_type == "Allowed Categories":
            allowed = st.text_input(
                "Enter allowed values (comma-separated)",
                key="val_allowed"
            )

        # ---- BUTTON ----
        check = st.button("Check Violations", key="val_button")

        if check:

            if validation_type == "Numeric Range":

                if col_val not in num_cols:
                    st.error("Selected column is not numeric")

                elif min_val > max_val:
                    st.error("Minimum cannot be greater than maximum")
                    st.stop()

                else:
                    violations = df[(df[col_val] < min_val) | (df[col_val] > max_val)]

            elif validation_type == "Allowed Categories":
                allowed_list = [x.strip() for x in allowed.split(",")]
                violations = df[~df[col_val].isin(allowed_list)]

            elif validation_type == "Non-null Constraint":
                violations = df[df[col_val].isnull()]

            # ---- SHOW RESULTS ----
            if not violations.empty:
                st.error(f"Violations found: {len(violations)}")
                st.dataframe(violations)

                csv = violations.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Violations CSV",
                    data=csv,
                    file_name="violations.csv",
                    mime="text/csv",
                    key="val_download"
                )
            else:
                st.success("No violations found")