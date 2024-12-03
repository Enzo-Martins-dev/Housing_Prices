from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px

df = pd.read_csv('housing_no_nulls_nor_outliers.csv')

app = Dash()

app.layout = [
    html.Div(children='My First App with Data'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=10),
    dcc.Graph(figure=px.histogram(df, x='median_income', y='median_house_value'))
]

if __name__ == '__main__':
    app.run(debug=True)
