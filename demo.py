import sqlite3
import pandas as pd

def return_df():
    # Create a connection to the database
    conn = sqlite3.connect('climate.db')

    # Create a cursor object
    c = conn.cursor()

    # Execute the SQL query
    c.execute("SELECT * FROM co2_and_energy LIMIT 5")

    # Fetch all the rows
    rows = c.fetchall()

    # Create a dataframe from the rows
    df = pd.DataFrame(rows, columns=['column1', 'column2', 'column3', 'column4', 'column5'])

    # Close the connection
    conn.close()

    # Return the dataframe
    return df

# Print the dataframe
print(return_df())
