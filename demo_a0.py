
import sqlite3
import pandas as pd

def return_df():
    # Create a connection to the database
    conn = sqlite3.connect('climate.db')

    # SQL query
    sql_query = """
    SELECT countries.Country, SUM(co2_and_energy.CO2_emission) as Total_CO2_Emissions 
    FROM co2_and_energy 
    JOIN countries ON co2_and_energy.Country_id = countries.id 
    WHERE co2_and_energy.Year = 2010 
    GROUP BY countries.Country
    """

    # Execute the query and return a dataframe
    df = pd.read_sql_query(sql_query, conn)

    # Close the connection
    conn.close()

    return df

# Print the dataframe
print(return_df())
