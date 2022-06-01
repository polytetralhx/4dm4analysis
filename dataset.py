import pandas as pd
import sqlite3

class Dataset():
    def __init__(self, db_dir):
        self.db_dir = db_dir
        self.db = sqlite3.connect(db_dir)
    
    def query(self, sql):
        """A method to query the SQLite Database with SQL and returns the pandas.DataFrame"""
        return pd.read_sql(sql, self.db)
    
    @staticmethod
    def format_where(where: dict):
        where_string = []
        for k, v in where.items():
            if isinstance(v, list):
                where_string.append(k + " IN " + str(tuple(v)))
                continue
            if not isinstance(v, str):
                v = str(v)
            if all([x not in v for x in ["=", ">", "<"]]):
                where_string.append(k + "=" + v)
                continue
            where_string.append(k + v)
        return " AND ".join(where_string)

    def select_script(self, table: str, columns: list = ["*"], where: dict = {}):
        sql = "SELECT {} FROM {}{}"
        col_string = ", ".join(columns)
        where_fmt = self.format_where(where)

        if where_fmt:
            return sql.format(col_string, table, " WHERE " + where_fmt)
        return sql.format(col_string, table, "")

    def select(self, table, columns: list = ["*"], where: dict = {}):
        """
        A method to select the data from table: **table**, returns all data from table if **columns** and **where** is not provided

        Example 1 : Select with conditional filtering
        ```
        # Select player_name, beatmap_type, beatmap_tag, score, score_logit from scores
        # where score > 990k in Qualifiers Round
        ds = dataset.select(
            table='scores', 
            columns=['player_name', 'beatmap_type', 'beatmap_tag', 'score', 'score_logit'], 
            where={
                'score': ">990000",
                'round': "\"Q\""
            }
        )
        ```

        Example 2 : Select without column provided
        ```
        # Select all columns from scores
        # where score > 990k
        ds = dataset.select(
            table='scores', 
            columns=['player_name', 'beatmap_type', 'beatmap_tag', 'score', 'score_logit'], 
            where={
                'score': ">990000"
            }
        )
        ```
        """
        return self.query(self.select_script(table, columns, where))
    
    def get_old_dataset(self, rounds: list, beatmap_types: list, logit: bool):
        where = {
            'round': rounds,
            'beatmap_type': beatmap_types
        }
        column = ['player_name', 'round || "_" || beatmap_type || "_" || beatmap_tag as column_name']
        if logit:
            column.append('score_logit')
        else:
            column.append('score')
        _data = self.select('scores', column, where)
        indecies = _data['player_name'].unique()
        columns = _data['column_name'].unique()
        new_dataframe = pd.DataFrame(index=indecies, columns=columns)
        for data in _data.values:
            player_name, column_name, score = data
            new_dataframe.at[player_name, column_name] = score
        return new_dataframe
    
    def get_label(self, rounds: list, beatmap_type: str, logit=False):
        """
        This is a method applied for implementing Linear Regression

        **rounds** (list) : the interested rounds (NOTE: It doesn't work with list with 1 length)
        **beatmap_type** (str) : a beatmap category (beatmap_type) that is interested
        **logit** (bool, default=False) : if this parameter is True, the method returns the average logit value of the score and if it is False, the method returns the average score
        """
        round_enumerate = {v: k for k, v in enumerate(rounds)}
        beatmap_type = f"\"{beatmap_type}\""
        where_fmt = self.format_where({'beatmap_type': beatmap_type, 'round': rounds})
        logit_str = "_logit" if logit else ""
        _data = self.query(f"SELECT player_name, round, avg(score{logit_str}) as avg_score{logit_str} FROM scores WHERE " + where_fmt + " GROUP BY player_name, round")
        _data['round_ord'] = _data['round'].apply(lambda x: round_enumerate.get(x))
        _data = _data.drop('round', axis=1)
        return _data
        

if __name__ == "__main__":
    _4dm4 = "4dm4.db"
    dataset = Dataset(_4dm4)
