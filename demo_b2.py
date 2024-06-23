
import sqlite3
import pandas as pd

def return_df():
    # Create a connection to the database
    conn = sqlite3.connect('climate.db')

    # SQL query
    query = "SELECT e.Energy_type, ce.Energy_consumption FROM co2_and_energy ce JOIN countries c ON ce.Country_id = c.id JOIN energy_type e ON ce.Energy_type_id = e.id WHERE c.Country = 'United States' AND ce.Year = 2000"

    # Execute the query and fetch the data into a dataframe
    df = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    # Return the dataframe
    return df

# Print the dataframe
print(return_df())
