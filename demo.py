import sqlite3
import pandas as pd

def return_df():
    # Create a connection to the database
    conn = sqlite3.connect('climate.db')

    # Define the SQL query
    query = "SELECT Year, SUM(CO2_emission) as Total_CO2_Emission FROM co2_and_energy GROUP BY Year ORDER BY Year"

    # Execute the query and store the result in a dataframe
    df = pd.read_sql_query(query, conn)

    # Close the connection to the database
    conn.close()

    # Return the dataframe
    return df

# Call the function and print the dataframe
df = return_df()
print(df)
