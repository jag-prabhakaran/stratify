
import sqlite3
import pandas as pd

def return_df():
    # Create a connection to the database
    conn = sqlite3.connect('climate.db')

    # SQL query
    query = "SELECT e.Energy_type, ce.Energy_production FROM co2_and_energy ce JOIN countries c ON ce.Country_id = c.id JOIN energy_type e ON ce.Energy_type_id = e.id WHERE c.Country = 'China' AND ce.Year = 2015"

    # Execute the query and return a dataframe
    df = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    return df

# Print the dataframe
print(return_df())
