import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

app = dash.Dash(__name__)

# External stylesheet for Google Fonts and custom CSS
external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?family=Calibri:wght@400;700&display=swap",
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

# Example list of figures with titles and insights
figures = [
    {
        'title': 'Line Plot',
        'figure': go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 1, 2], mode='lines')),
        'insight': 'This is a line plot showing a decreasing trend.',
    },
    {
        'title': 'Bar Chart',
        'figure': go.Figure(data=go.Bar(x=[1, 2, 3], y=[2, 5, 3])),
        'insight': 'This bar chart shows that the second category has the highest value.',
    },
    {
        'title': 'Pie Chart',
        'figure': go.Figure(data=go.Pie(labels=['A', 'B', 'C'], values=[30, 50, 20])),
        'insight': 'This pie chart shows that category B is the largest.',
    },
    {
        'title': 'Scatter Plot',
        'figure': go.Figure(data=go.Scatter(x=[1, 2, 3], y=[2, 4, 5], mode='markers')),
        'insight': 'This scatter plot shows a positive correlation.',
    }
]

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

@app.callback(
    Output('graph-container', 'children'),
    Input('graph-container', 'id')
)
def update_graph_container(_):
    graph_boxes = []
    for item in figures:
        graph_box = html.Div([
            html.H3(item['title'], className='graph-title'),
            html.Div([
                html.Div(dcc.Graph(figure=item['figure']), className='graph'),
                html.Div(item['insight'], className='insight')
            ], className='graph-insight')
        ],
            className='graph-box'
        )
        graph_boxes.append(graph_box)
    return graph_boxes

if __name__ == '__main__':
    app.run_server(debug=True)