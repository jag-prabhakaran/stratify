import sqlite3
import pandas as pd

def return_df():
    # Connect to the database
    conn = sqlite3.connect('climate.db')

    # SQL query
    query = "SELECT co2_and_energy.Year, co2_and_energy.Energy_consumption FROM co2_and_energy JOIN countries ON co2_and_energy.Country_id = countries.id WHERE countries.Country = 'India' AND co2_and_energy.Year BETWEEN 2000 AND 2020 ORDER BY co2_and_energy.Year"

    # Execute the query and create a dataframe
    df = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    # Return the dataframe
    return df

# Print the dataframe
print(return_df())
