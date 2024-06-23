
import sqlite3
import pandas as pd

def return_df():
    # Create a connection to the database
    conn = sqlite3.connect('climate.db')

    # SQL query
    query = """
    SELECT t.Year, t.Anomaly, SUM(c.CO2_emission) as Total_CO2_emission 
    FROM temperatures t 
    JOIN co2_and_energy c 
    ON t.Year = c.Year 
    WHERE t.Year BETWEEN 1960 AND 2010 
    GROUP BY t.Year, t.Anomaly 
    ORDER BY t.Year
    """

    # Execute the query and return a dataframe
    df = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    return df

# Print the dataframe
print(return_df())
