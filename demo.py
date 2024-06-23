import sqlite3
import pandas as pd

def return_df():
    # Create a connection to the database
    conn = sqlite3.connect('climate.db')

    # SQL query
    query = "SELECT Year, GDP FROM co2_and_energy JOIN countries ON co2_and_energy.Country_id = countries.id WHERE countries.Country = 'India' AND Year BETWEEN 1990 AND 2010 ORDER BY Year"

    # Execute the query and return a dataframe
    df = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    return df

# Print the dataframe
print(return_df())
