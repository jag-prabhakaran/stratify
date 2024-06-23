import plotly.graph_objects as go

def bagel_tron(dfs_df):
    figures = [
        {
            'title': 'CO2 Emissions Density by Country',
            'figure': go.Figure(data=go.Choropleth(
                locations=dfs_df[0]['Country'], 
                locationmode='country names', 
                z=dfs_df[0]['Total_CO2_Emission'], 
                colorscale='Plasma', 
                colorbar_title='CO2 Emissions',
            )).update_layout(
                title='CO2 Emissions Density by Country', 
                geo=dict(showframe=False, showcoastlines=False, projection_type='equirectangular'), 
                height=600, 
                margin={"r":0,"t":0,"l":0,"b":0}
            ),
            'insight': 'This map shows the total CO2 emissions for each country in the year 2010.',
            'is_map': True
        },
        {
            'title': 'Global Temperature Anomaly vs CO2 Emissions',
            'figure': go.Figure(data=go.Scatter(
                x=dfs_df[1]['Year'], 
                y=dfs_df[1]['Anomaly'], 
                mode='markers',
                marker=dict(
                    size=dfs_df[1]['Total_CO2_emission'],
                    sizemode='area',
                    sizeref=2.*max(dfs_df[1]['Total_CO2_emission'])/(40.**2),
                    sizemin=4
                ),
                name='CO2 Emissions'
            )).update_layout(
                title='Global Temperature Anomaly vs CO2 Emissions',
                xaxis_title='Year',
                yaxis_title='Temperature Anomaly',
                showlegend=True
            ),
            'insight': 'This scatter plot shows the correlation between the change in global temperature anomaly and the total global CO2 emissions from 1960 to 2010.',
            'is_map': False
        },
        {
            'title': 'Energy Consumption by Type in the US (2000)',
            'figure': go.Figure(data=go.Bar(
                x=dfs_df[2]['Energy_type'], 
                y=dfs_df[2]['Energy_consumption']
            )).update_layout(
                title='Energy Consumption by Type in the US (2000)',
                xaxis_title='Energy Type',
                yaxis_title='Energy Consumption'
            ),
            'insight': 'This bar graph shows the distribution of energy consumption by type for the United States in the year 2000.',
            'is_map': False
        },
        {
            'title': 'Energy Production by Type in China (2015)',
            'figure': go.Figure(data=go.Pie(
                labels=dfs_df[3]['Energy_type'], 
                values=dfs_df[3]['Energy_production']
            )).update_layout(
                title='Energy Production by Type in China (2015)'
            ),
            'insight': 'This pie chart shows the proportion of energy production by type for China in the year 2015.',
            'is_map': False
        },
        {
            'title': 'Energy Intensity per Capita in India (1990-2020)',
            'figure': go.Figure(data=go.Scatter(
                x=dfs_df[4]['Year'], 
                y=dfs_df[4]['Energy_intensity_per_capita'], 
                mode='lines'
            )).update_layout(
                title='Energy Intensity per Capita in India (1990-2020)',
                xaxis_title='Year',
                yaxis_title='Energy Intensity per Capita'
            ),
            'insight': 'This line graph shows how the energy intensity per capita has changed over time for India from 1990 to 2020.',
            'is_map': False
        }
    ]
    return figures
