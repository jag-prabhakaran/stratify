
import sqlite3
import pandas as pd

def return_df():
    # Create a connection to the database
    conn = sqlite3.connect('climate.db')

    # SQL query
    query = "SELECT c.Country, SUM(e.CO2_emission) as Total_CO2_Emission FROM co2_and_energy e JOIN countries c ON e.Country_id = c.id WHERE e.Year = 2010 GROUP BY c.Country"

    # Execute the query and return a dataframe
    df = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    return df

# Print the dataframe
print(return_df())
