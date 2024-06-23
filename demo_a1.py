
import sqlite3
import pandas as pd

def return_df():
    # Create a connection to the database
    conn = sqlite3.connect('climate.db')

    # SQL query
    query = "SELECT Year, Anomaly FROM temperatures WHERE Year BETWEEN 1900 AND 2000"

    # Execute the query and store the result in a DataFrame
    df = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    # Return the DataFrame
    return df

# Call the function and print the DataFrame
df = return_df()
print(df)
