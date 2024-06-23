
import sqlite3
import pandas as pd

def return_df():
    # Connect to the database
    conn = sqlite3.connect('climate.db')

    # SQL query
    sql_query = "SELECT Year, GDP FROM co2_and_energy WHERE Country_id = (SELECT id FROM countries WHERE Country = 'India') AND Year BETWEEN 1990 AND 2010 ORDER BY Year"

    # Execute the query and store the result in a dataframe
    df = pd.read_sql_query(sql_query, conn)

    # Close the connection
    conn.close()

    # Return the dataframe
    return df

# Print the dataframe
print(return_df())
