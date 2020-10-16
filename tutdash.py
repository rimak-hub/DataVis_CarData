#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    return  dcc.Markdown('''
Data Desciption:

             Year       Selling_Price     Present_Price       Kms_Driven       Owner

count    301.000000      301.000000     301.000000          301.000000      301.000000

mean   2013.627907        4.661296       7.628472          36947.205980      0.043189

std       2.891554       5.082812       8.644115           38886.883882       0.247915

min    2003.000000       0.100000       0.320000           500.000000        0.000000

25%    2012.000000       0.900000       1.200000          15000.000000       0.000000

50%    2014.000000       3.600000       6.400000         32000.000000       0.000000

75%    2016.000000       6.000000       9.900000         48767.000000       0.000000

max    2018.000000      35.000000      92.600000        500000.000000       3.000000


''')








@app.callback(Output('subtab2', 'children'),
              [Input('tab1', 'value')])
def update_val(value):
    return 'Les variables selling_price , years, kms_Driven et Transmission sont corrélées'

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
score SVR linear kernel:  0.04186029418609705

score sklearn linear regression:  0.0557625625111281

Numpy linear regression parameters:  [ 4.15091695e-01 -8.31178925e+02]

Scipy Linear regression parameters:  0.4150916946355396 -831.1789245913283 0.2361409801604273 3.495472434809122e-05 0.0987819729376285


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
Multivariate linear regression sklearn

input variables: year, Kms_driven, Transmission 

score: 0.20309987058925372

Intercept: -1081.0946700995719

Coefficients: [5.38542399e-01 1.74728571e-05 5.16506926e+00]

Conclusion: la regression multivariée a un meilleur score que la regression lineaire simple et ainsi elle est plus adaptée à notre problème vu que le selling price et l'age de la voiture ne sont pas linerement coréllés.

''')


   
if __name__ == '__main__':
    app.run_server(debug=True)
