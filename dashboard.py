from dash import Dash, html, dash_table, dcc, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Carregar os dados pré-processados
df = pd.read_csv('./datasets/housing_clean.csv')

# Inicializar o app Dash
app = Dash(__name__)

# Layout do dashboard
app.layout = html.Div([
    html.H1("Dashboard de Dados de Imóveis", style={'textAlign': 'center'}),

    # Filtro por Proximidade ao Oceano
    html.Div([
        html.Label("Filtrar por Proximidade ao Oceano:"),
        dcc.Dropdown(
            id='ocean_proximity_filter',
            options=[{'label': op, 'value': op} for op in df['ocean_proximity'].unique()],
            value=None,
            multi=True,
            placeholder="Selecione a proximidade..."
        ),
    ], style={'width': '40%', 'margin': 'auto'}),

    # Filtro por Faixa de Renda Mediana com intervalos de 10k em 10k
    html.Div([
        html.Label("Filtrar por Faixa de Renda Mediana (10k em 10k):"),
        dcc.RangeSlider(
            id='income_filter',
            min=0,
            max=10,  # Usaremos a faixa de 0 a 10 para representar 0k a 100k
            step=1,
            marks={i: f"{i*10}k" for i in range(11)},
            value=[0, 10]
        )
    ], style={'width': '60%', 'margin': 'auto'}),

    # Tabela de Dados
    html.Br(),
    dash_table.DataTable(
        id='housing_table',
        columns=[{"name": col, "id": col} for col in df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_data={'whiteSpace': 'normal', 'height': 'auto'},
        style_cell={'textAlign': 'left'}
    ),

    # Gráficos
    html.Br(),
    dcc.Graph(id='scatter_plot'),
    html.Br(),
    dcc.Graph(id='histogram'),
    html.Br(),
    dcc.Graph(id='box_plot'),
    html.Br(),
    dcc.Graph(id='correlation_heatmap'),
    
    # Resultados de Regressão Linear
    html.Br(),
    html.Div(id='regression_results', style={'textAlign': 'center', 'fontSize': '20px', 'margin': '20px'}),

    # Insights
    html.Br(),
    html.Div(id='insights', style={'textAlign': 'center', 'fontSize': '20px', 'margin': '20px'})
])

# Callback para atualizar a tabela e gráficos
@app.callback(
    [Output('housing_table', 'data'),
     Output('scatter_plot', 'figure'),
     Output('histogram', 'figure'),
     Output('box_plot', 'figure'),
     Output('correlation_heatmap', 'figure'),
     Output('regression_results', 'children'),
     Output('insights', 'children')],
    [Input('ocean_proximity_filter', 'value'),
     Input('income_filter', 'value')]
)
def update_dashboard(ocean_proximity_filter, income_filter):
    # Filtrar os dados com base nos filtros aplicados
    filtered_df = df.copy()
    
    # Filtro de proximidade ao oceano
    if ocean_proximity_filter:
        filtered_df = filtered_df[filtered_df['ocean_proximity'].isin(ocean_proximity_filter)]
    
    # Filtro de faixa de renda (em intervalos de 10k)
    min_income = income_filter[0] * 10000
    max_income = income_filter[1] * 10000
    filtered_df = filtered_df[(filtered_df['median_income'] >= min_income) & 
                               (filtered_df['median_income'] <= max_income)]

    # Tabela de Dados
    table_data = filtered_df.to_dict('records')

    # Gráfico de Dispersão
    scatter_fig = px.scatter(
        filtered_df, 
        x='median_income', 
        y='median_house_value', 
        color='ocean_proximity',
        title='Renda vs. Valor da Casa',
        labels={'median_income': 'Renda Mediana', 'median_house_value': 'Valor da Casa'}
    )

    # Histograma
    histogram_fig = px.histogram(
        filtered_df,
        x='median_income',
        y='median_house_value',
        color='ocean_proximity',
        title='Distribuição do Valor da Casa por Renda',
        labels={'median_income': 'Renda Mediana', 'median_house_value': 'Valor da Casa'},
        barmode='group'
    )

    # Box plot
    box_plot_fig = px.box(
        filtered_df,
        x='ocean_proximity',
        y='median_house_value',
        color='ocean_proximity',
        title='Valor da Casa por Proximidade ao Oceano',
        labels={'ocean_proximity': 'Proximidade ao Oceano', 'median_house_value': 'Valor da Casa'}
    )

    # Heatmap de Correlação entre variáveis numéricas
    numeric_cols = filtered_df.select_dtypes(include=[np.number])
    corr_matrix = numeric_cols.corr()
    correlation_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
        colorbar=dict(title='Correlação')
    ))
    correlation_heatmap.update_layout(
        title="Correlação entre as Variáveis",
        xaxis_title="Variáveis",
        yaxis_title="Variáveis"
    )

    # Regressão Linear: Predição do valor da casa baseado na renda
    X = filtered_df[['median_income']]
    y = filtered_df['median_house_value']
    
    # Divisão de dados para treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Criando e treinando o modelo de regressão linear
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predições
    y_pred = model.predict(X_test)

    # Avaliação do modelo
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Gráfico de Dispersão Real vs Predito
    regression_scatter = go.Figure()
    regression_scatter.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode='markers', name='Predições',
        marker=dict(color='blue')
    ))
    regression_scatter.update_layout(
        title="Valores Reais vs Preditos",
        xaxis_title="Valor Real da Casa",
        yaxis_title="Valor Predito da Casa"
    )

    # Gráfico de Resíduos
    residuals = y_test - y_pred
    residuals_fig = go.Figure()
    residuals_fig.add_trace(go.Scatter(
        x=y_test, y=residuals, mode='markers', name='Resíduos',
        marker=dict(color='red')
    ))
    residuals_fig.update_layout(
        title="Resíduos da Regressão Linear",
        xaxis_title="Valor Real da Casa",
        yaxis_title="Resíduos"
    )

    # Exibição dos Resultados da Regressão Linear
    regression_results = f"""
    Regressão Linear:
    MAE (Erro Absoluto Médio): ${mae:.2f}
    MSE (Erro Quadrático Médio): ${mse:.2f}
    RMSE (Raiz do Erro Quadrático Médio): ${rmse:.2f}
    """

    # Insights
    avg_income = filtered_df['median_income'].mean()
    avg_house_value = filtered_df['median_house_value'].mean()
    total_houses = filtered_df.shape[0]
    insight_text = (
        f"Renda Média: ${avg_income:.2f} | "
        f"Valor Médio da Casa: ${avg_house_value:.2f} | "
        f"Total de Casas: {total_houses}"
    )

    return table_data, scatter_fig, histogram_fig, box_plot_fig, correlation_heatmap, regression_results, insight_text

# Executar o servidor
if __name__ == '__main__':
    app.run_server(debug=True)
