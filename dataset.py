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

if __name__ == "__main__":
    _4dm4 = "4dm4.db"
    dataset = Dataset(_4dm4)
    
    ds = dataset.select(
        table='scores'
    )

    print(ds)