import pandas as pd
import numpy as np

class MacroDataMerger:
    def __init__(self, file_path, start_date=None):
        self.file_path = file_path
        self.excel_file = pd.ExcelFile(file_path)
        self.dataframes = {}
        self.base_df = None
        self.start_date = pd.to_datetime(start_date) if start_date else None

    def load_data(self):
        """Load all sheets into individual DataFrames."""
        for sheet in self.excel_file.sheet_names:
            df = self.excel_file.parse(sheet)
            df.columns = [col.strip() for col in df.columns]
            df.rename(columns={df.columns[0]: "Date", df.columns[1]: sheet}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"])
            df.sort_values("Date", inplace=True)
            self.dataframes[sheet] = df

        if self.start_date:
            for sheet in self.dataframes:
                self.dataframes[sheet] = self.dataframes[sheet][self.dataframes[sheet]["Date"] >= self.start_date]

    def set_base_df(self, base_sheet="Unemployment"):
        """Set the base DataFrame (e.g., quarterly unemployment data)."""
        self.base_df = self.dataframes[base_sheet].copy()
        self.base_df.set_index("Date", inplace=True)

    def merge_all(self):
        """Merge all datasets using forward fill only if within 1 month tolerance."""
        self.load_data()
        self.set_base_df()
        
        # Initialize merged DataFrame with the base DataFrame
        merged_df = self.base_df.copy()

        for sheet, df in self.dataframes.items():
            if sheet == "Unemployment":
                continue
            df_sorted = df.sort_values("Date")
            merged_df = pd.merge_asof(
                merged_df.sort_values("Date"),
                df_sorted,
                on="Date",
                direction="forward",
                tolerance=pd.Timedelta("31D")
            )

        return merged_df

class WRDSFundamentalsDataMerger:

    def __init__(self, file_path):
        self.file_path = file_path
        self.sheet_names = pd.ExcelFile(file_path).sheet_names
        self.raw_data = self._load_data()
        self.restructured_data = self._restructure_data()

    def _load_data(self):
        """Loads each sheet and combines into a multi-index dataframe (feature, time, company)"""
        data_dict = {}
        for sheet in self.sheet_names:
            df = pd.read_excel(self.file_path, sheet_name=sheet)
            df['calendar_quarter'] = pd.to_datetime(df['calendar_quarter'])
            df.set_index('calendar_quarter', inplace=True)
            data_dict[sheet] = df
        # Concat into multi-index frame: (feature, calendar_quarter)
        panel_df = pd.concat(data_dict, names=['feature', 'calendar_quarter'])
        return panel_df

    def _restructure_data(self):
        """Transforms into a dict of DataFrames, one per company"""
        company_dict = {}
        # Iterate over company tickers (columns)
        for company in self.raw_data.columns:
            df_company = self.raw_data.xs(key=company, axis=1)
            # Pivot so index = time, columns = features
            df_company = df_company.unstack(level=0)  # now rows = date, cols = features
            company_dict[company] = df_company
        return company_dict

    def get_company_data(self, ticker):
        """Returns a 2D DataFrame: rows = dates, columns = features"""
        return self.restructured_data.get(ticker, pd.DataFrame())
    
    @staticmethod
    def get_tickers_from_file(file_path):
        """Extracts tickers from a file, assuming the first column contains them."""
        df = pd.read_excel(file_path, header=0)
        tickers = df.columns.tolist()[1:]
        if ':' in tickers[0]:
            tickers = [ticker.split(':')[0] for ticker in tickers]
        else:
            return tickers

# Example usage
# file_path = "data/WRDS_features_spx.xlsx"
# wrds = WRDSFundamentalsDataMerger(file_path)
# aapl_data = wrds.get_company_data("AAPL")  # Example: Get AAPL fundamentals
# print(aapl_data.head())

class CreditRatingDataMerger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_excel(file_path)

    def get_df(self):
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.clean_invalid_companies(self.df)
        self.df = self.transform_data()
        return self.df
    
    # only single sheet
    def transform_data(self):
        """Transform the data to a more usable format."""
        # Assuming the first column is 'Date' and the rest are ratings
        rating_map = {
            # Standard ratings
            'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18, 'A+': 17, 'A': 16, 'A-': 15,
            'BBB+': 14, 'BBB': 13, 'BBB-': 12, 'BB+': 11, 'BB': 10, 'BB-': 9,
            'B+': 8, 'B': 7, 'B-': 6, 'CCC+': 5, 'CCC': 4, 'CCC-': 3, 'CC': 2, 'C': 1,
            'SD': 0, 'D': 0, 'NR': np.nan, '': np.nan,
            
            # "pi" ratings (public information)
            'AAApi': 21, 'AA+pi': 20, 'AApi': 19, 'AA-pi': 18, 'A+pi': 17, 'Api': 16, 'A-pi': 15,
            'BBB+pi': 14, 'BBBpi': 13, 'BBB-pi': 12, 'BB+pi': 11, 'BBpi': 10, 'BB-pi': 9,
            'B+pi': 8, 'Bpi': 7, 'B-pi': 6, 'CCC+pi': 5, 'CCCpi': 4, 'CCC-pi': 3,
        }

        def map_with_fallback(col):
            if col.name != self.df.columns[0] and col.name != self.df.index.name and col.dtype == 'object':
                # First ensure NaN values don't cause issues
                temp_col = col.copy()
                # Map values and handle NaN explicitly
                return temp_col.map(lambda x: rating_map.get(x, np.nan) if pd.notna(x) else np.nan)
            return col

        return self.df.apply(map_with_fallback)
    
    def clean_invalid_companies(self,df):
        """
        Remove columns (companies) with invalid IDs from the DataFrame.
        """
        # Find columns containing invalid company IDs
        invalid_cols = []
        
        for col in df.columns:
            if (df[col] == '#INVALID COMPANY ID').any():
                invalid_cols.append(col)
        if invalid_cols:
            print(f"Removing {len(invalid_cols)} invalid company columns: {invalid_cols}")
            
        # Drop the invalid columns and return the cleaned DataFrame
        return df.drop(columns=invalid_cols)
    
# file_path = "data/all_companies_historical_credit_rating_sp_capital_iq_pro_2.xlsx"
# credit_rating_data = CreditRatingDataMerger(file_path)
# credit_rating_df = credit_rating_data.get_df()
# print(credit_rating_df.head())
