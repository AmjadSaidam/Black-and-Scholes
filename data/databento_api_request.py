import databento as db # conda install databento (pip show <package name> to check install)
import pandas as pd

class DatabentoAsset():
    """
    Intended for pulling underlying and option data, with alligned end dates
    underlying start date may be lesser for parameter estimation stability
    """
    def __init__(self, api_key: str, equity: str, end_date: str):
        self.api_key = api_key
        self.equity = equity
        self.end_date = end_date
        self.client = db.Historical(self.api_key)

    def option_equity_schema(self, start_date) -> pd.DataFrame:
        """
        L3 option data for equity 
        """
        option_symbol = self.equity + '.OPT'
        data = self.client.timeseries.get_range(
            dataset = 'OPRA.PILLAR', # OPRA options data provider 
            schema = 'definition', # data schema 
            stype_in = 'parent', # to get chain data 
            symbols = [option_symbol], # option data 
            start = start_date, 
            end = self.end_date
        )

        return data.to_df()

    def equity_schema(self, start_date, schema: str = 'OHLCV-1D'):
        """
        schema for equity (underlying)
        """
        data = self.client.timeseries.get_range(
            dataset = 'XNAS.ITCH', # Nasdaq Total View data provider
            schema = schema, 
            symbols = self.equity, 
            start = start_date, 
            end = self.end_date
        )
        
        return data.to_df()
    