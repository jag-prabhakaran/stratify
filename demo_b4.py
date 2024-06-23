
import sqlite3
import pandas as pd

def return_df():
    # Create a connection to the database
    conn = sqlite3.connect('climate.db')

    # SQL query
    sql_query = "SELECT Year, Energy_intensity_per_capita FROM co2_and_energy JOIN countries ON co2_and_energy.Country_id = countries.id WHERE countries.Country = 'India' AND Year BETWEEN 1990 AND 2020 ORDER BY Year"

    # Execute the query and store the result in a dataframe
    df = pd.read_sql_query(sql_query, conn)

    # Close the connection
    conn.close()

    # Return the dataframe
    return df

# Print the dataframe
print(return_df())
