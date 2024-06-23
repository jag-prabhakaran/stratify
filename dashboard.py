import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from flask import Flask, redirect, url_for, render_template_string

app = dash.Dash(__name__)

# External stylesheet for Google Fonts and custom CSS
external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?family=Proxima+Nova&display=swap",
        "rel": "stylesheet",
    },
    {
        "href": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css",
        "rel": "stylesheet",
    },
    {
        "href": "https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css",
        "rel": "stylesheet",
    },
    "/static/css/styles.css",
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Flask server instance
server = app.server

# Example list of figures with titles and insights
figures = [
    {
        'title': 'Line Plot',
        'figure': go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 1, 2], mode='lines')),
        'insight': 'This is a line plot showing a decreasing trend.',
        'is_map': False
    },
    {
        'title': 'Bar Chart',
        'figure': go.Figure(data=go.Bar(x=[1, 2, 3], y=[2, 5, 3])),
        'insight': 'This bar chart shows that the second category has the highest value.',
        'is_map': False
    },
    {
        'title': 'Pie Chart',
        'figure': go.Figure(data=go.Pie(labels=['A', 'B', 'C'], values=[30, 50, 20])),
        'insight': 'This pie chart shows that category B is the largest.',
        'is_map': False
    },
    {
        'title': 'Scatter Plot',
        'figure': go.Figure(data=go.Scatter(x=[1, 2, 3], y=[2, 4, 5], mode='markers')),
        'insight': 'This scatter plot shows a positive correlation.',
        'is_map': False
    },
    {
        'title': 'Map Plot',
        'figure': go.Figure(data=go.Scattergeo(lon=[-75, -80, -70], lat=[45, 50, 40], mode='markers')).update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0}),
        'insight': 'This is a map plot showing geographic locations.',
        'is_map': True
    }
]

# Sort figures with map type first
figures.sort(key=lambda x: not x['is_map'])

# Layout for the main dashboard
app.layout = html.Div([
    html.H1("Dashboard with Sidebar and Dynamic Graph Boxes", className='main-title'),
    html.Div([
        # Sidebar
        html.Div([
            html.Ul([
                html.Li(dcc.Link("Chatbot", href="#", className='sidebar-item')),
                html.Li(dcc.Link("Ad Hoc", href="#", className='sidebar-item')),
                html.Li(dcc.Link("Dashboards", href="#", className='sidebar-item'))
            ], className='sidebar-list')
        ], className='sidebar'),

        # Main content
        html.Div(id='graph-container', className='main-content')
    ], className='dashboard-container')
])

# Callback to update graph container
@app.callback(
    Output('graph-container', 'children'),
    Input('graph-container', 'id')
)
def update_graph_container(_):
    graph_boxes = []
    for item in figures:
        graph_box = html.Div([
            html.A([
                html.H3(item['title'], className='graph-title'),
            ], href=f"/graph/{item['title']}"),
            html.Div(dcc.Graph(figure=item['figure']), className='graph')
        ],
            className='graph-box double-width' if item.get('is_map') else 'graph-box'
        )
        graph_boxes.append(graph_box)
    return graph_boxes

# Flask route for larger graph view
@server.route('/graph/<title>')
def graph_page(title):
    # Find the corresponding figure
    figure_data = next((item for item in figures if item['title'] == title), None)
    if not figure_data:
        return "Graph not found", 404
    
    figure = figure_data['figure']
    insight = figure_data['insight']
    is_map = figure_data['is_map']
    
    # Render the graph page using Plotly's JavaScript library
    graph_html = figure.to_html(full_html=False, include_plotlyjs='cdn')
    
    return render_template_string("""
        <html>
        <head>
            <title>{{ title }}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: 'Proxima Nova', sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 0;
                }
                .main-title {
                    font-size: 2.5rem;
                    text-align: center;
                    padding: 20px 0;
                    margin-bottom: 20px;
                }
                .dashboard-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    min-height: 100vh;
                }
                .sidebar {
                    width: 20%;
                    background-color: #2c3e50;
                    padding: 20px;
                    border-radius: 5px;
                    height: 100%;
                    overflow-y: auto;
                    box-shadow: 2px 0 5px rgba(0,0,0,0.1);
                    position: fixed;
                    top: 0;
                    left: 0;
                }
                .sidebar-list {
                    list-style-type: none;
                    padding: 0;
                }
                .sidebar-item {
                    color: white;
                    text-decoration: none;
                    font-size: 1.5rem;
                    padding: 15px 0;
                }
                .sidebar-item:hover {
                    background-color: #1a252f;
                }
                .main-content {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr); /* Two boxes per line */
                    gap: 20px; /* Spacing between graph boxes */
                    width: 80%; /* Adjusted width to accommodate the sidebar */
                    margin-left: 20%; /* Adjusted margin to accommodate the sidebar */
                    padding: 20px; /* Padding around the main content */
                }
                .graph-box {
                    border: 2px solid #3498db;
                    padding: 20px;
                    margin: 10px;
                    border-radius: 10px; /* Rounded corners */
                    background-color: white;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    transition: transform 0.2s ease-in-out;
                }
                .graph-box:hover {
                    transform: scale(1.02);
                }
                .double-width {
                    grid-column: span 2;
                }
                .graph {
                    width: 100%; /* Adjusted width to fill the box */
                }
                .graph-title {
                    text-align: center;
                    font-size: 1.8rem;
                    margin-bottom: 15px;
                }
                .back-button {
                    display: inline-block;
                    padding: 10px 20px;
                    background-color: #3498db;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    margin-top: 20px;
                }
                .back-button:hover {
                    background-color: #1e6fa8;
                }
            </style>
        </head>
        <body>
            <div class="graph-box-large">
                <h2 style="text-align: center;">{{ title }}</h2>
                <div style="width: 100%; height: 80vh;">{{ graph_html | safe }}</div>
                <a href="/" class="back-button">Back to Dashboard</a>
            </div>
        </body>
        </html>
    """, title=title, graph_html=graph_html)

if __name__ == '__main__':
    app.run_server(debug=True)