import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas_datareader.data as web
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import dash_table
import dash_table_experiments as dt
app = dash.Dash()



# Create a Dash layout
app.layout = html.Div([
    html.Div(
        html.H1('My Dashboard')
    ),
    dcc.Tabs(id='tabs', value='Tab1', children=[
        dcc.Tab(label='Exploration de Données', id='tab1', value='Tab1', children =[ dcc.Tabs(id="subtabs1", value='Subtab1', children=[
            dcc.Tab(label='Statistique', id='subtab1', value='Subtab1'),
            dcc.Tab(label='Correlation', id='subtab2', value='Subtab2'),
        ])
        ]),
        dcc.Tab(label='Regression Lineaire', id='tab2', value='Tab2', children=[dcc.Tabs(id="subtabs2", value='Subtab4', children=[
            dcc.Tab(label='Simple', id='subtab4', value='Subtab4'),
            dcc.Tab(label='Multivariée', id='subtab5', value='Subtab5')
        ]),
        ])
    ])
])

@app.callback(Output('subtab1', 'children'),
              [Input('tab1', 'value')])
def update_val(value):
    #df = pd.read_csv("carData.csv")

    #description= df.describe(include= "all")
    return  'blabla'






@app.callback(Output('subtab2', 'children'),
              [Input('tab1', 'value')])
def update_val(value):
    return '44'

@app.callback(
    Output('subtab4', 'children'),
              [Input('tab2', 'value')]
)
def update_value(input_data):
    #df = pd.read_csv("carData.csv")
    df = pd.read_csv("out.csv")

    #print(input_data)
    #return input_data
    return dcc.Graph(
           id='example-graph',
           figure={
                'data':[
                    go.Scatter(
                    x=df['Year'], 
                        y= df['Selling_Price'],
                         mode='markers',
                         opacity=0.8,
                         name= 'Data',
                         marker={
                             'size': 15,
                             'line': {'width': 0.5, 'color': 'white'}    },
                         
                     ),
                    go.Scatter(
                        x=df['Year'], 
                        y= df['Y_np'],
                        name= 'numpy LR', 
                        opacity=0.8,
                        line=dict(width= 2, color= 'rgb(238, 64, 53)')
                     ),
                    go.Scatter(
                        x=df['Year'], 
                        y= df['Y_Scipy'],
                        name= 'Scipy LR',
                        opacity=0.8,
                        line=dict(width= 2, color= 'rgb(123, 192, 67)')
                         
                     ),
                    go.Scatter(
                        x=df['Year'], 
                        y= df['Y_sklearnSLR'],
                        opacity=0.8,
                        name= 'Sklearn LR',
                        line=dict(width= 2, color= 'rgb(3, 146, 207)') ),

                    go.Scatter(
                        x=df['Year'], 
                        y= df['Y_svr'],
                        opacity=0.8,
                        name= 'SVR LR',
                        line=dict(width= 2, color= 'rgb(243, 119, 54)') )

                    
                 ],
                'layout': go.Layout(
                xaxis={'title': 'Year'},
                yaxis={'title': 'Selling Price'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0.0, 'y': 1},
                hovermode='closest'
            )
            }
        ), dcc.Markdown('''

Dash supports [Markdown](http://commonmark.org/help).

Markdown is a simple way to write and format text.
It includes a syntax for things like **bold text** and *italics*,
[links](http://commonmark.org/help), inline `code` snippets, lists,
quotes, and more.
''')


@app.callback(Output('subtab5', 'children'),
              [Input('tab2', 'value')])
def update_value(input_data):
    #df = pd.read_csv("carData.csv")
    df = pd.read_csv("out.csv")

    #print(input_data)
    #return input_data
    return dcc.Graph(
           id='example-graph',
           figure={
                'data':[
                    go.Scatter(
                        x=df['Year'], 
                        y= df['Selling_Price'],
                        mode='markers',
                        opacity=0.8,
                        name='Data',
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}    },
                         
                    ),
                    go.Scatter(
                        x=df['Year'], 
                        y= df['Y_sklearnMLR'],
                        mode='markers',
                        opacity=0.8,
                        name= 'sklearn MLR',
                         marker={
                             'size': 15,
                             'line': {'width': 0.5, 'color': 'white'}    },
                         
                     ), 
                    
                    
                 ],
                'layout': go.Layout(
                xaxis={'title': 'Year'},
                yaxis={'title': 'Selling Price'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0.0, 'y': 1},
                hovermode='closest'
            )
            }
        ),dcc.Markdown('''

Dash supports [Markdown](http://commonmark.org/help).

Markdown is a simple way to write and format text.
It includes a syntax for things like **bold text** and *italics*,
[links](http://commonmark.org/help), inline `code` snippets, lists,
quotes, and more.
''')


   
if __name__ == '__main__':
    app.run_server(debug=True)
