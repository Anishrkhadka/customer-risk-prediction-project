import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy.stats import gaussian_kde
import numpy as np

# color palette format
colour = ['#2a9d8f', '#003049', '#d62828', '#f77f00', '#fcbf49', '#eae2b7']

def plot_missing_data(df):
    nulls = df.isnull()
    fig = px.imshow(
        nulls.T,
        labels=dict(x="Rows", y="Columns", color="Missing"),
        color_continuous_scale=["#ffffff", colour[2]],
        aspect="auto",
    )
    fig.update_layout(title="Missing Values Heatmap", height=400)
    fig.show()

def plot_categorical_counts(df, columns, top_n=None, orient='v', title=f"Count of", category_color_map=None):
    for col in columns:
        data = df[col].value_counts().nlargest(top_n) if top_n else df[col].value_counts()
        data = data.reset_index().copy()
        data.columns = [col, 'count']

        if category_color_map:
            data['color'] = data[col].map(category_color_map)
            palette = data['color'].tolist()
        else:
            palette = colour

        if orient == 'h':
            fig = px.bar(
                data,
                x='count',
                y=col,
                orientation='h',
                text='count',
                color=col,
                color_discrete_sequence=palette,
                title=title
            )
        else:
            fig = px.bar(
                data,
                x=col,
                y='count',
                text='count',
                color=col,
                color_discrete_sequence=palette,
                title=title
            )
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        fig.show()


def plot_treemap(
    df,
    category_col,
    value_col=None,
    title="Treemap",
    category_palette=colour
):
    """
    Plot a treemap showing either:
    - counts of each category (if value_col is None), or
    - sum of a value column grouped by category.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        category_col (str): Column to group by (categorical).
        value_col (str, optional): If provided, will sum this column per category.
        title (str): Plot title.
        palette (dict, optional): Color mapping for categories (category: color hex).
    """
    if value_col:
        data = df.groupby(category_col)[value_col].sum().reset_index()
    else:
        data = df[category_col].value_counts().reset_index()
        data.columns = [category_col, 'count']
        value_col = 'count'

    # Add percent labels
    total = data[value_col].sum()
    data['percentage'] = (data[value_col] / total * 100).round(1)
    data['label'] = data[category_col].astype(str) + ' (' + data['percentage'].astype(str) + '%)'

    fig = px.treemap(
        data,
        path=['label'],
        values=value_col,
        color=category_col,
        color_discrete_map=category_palette,
        title=title
    )
    fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
    fig.show()



def plot_numerical_distribution(df, col, title=None, nbins=30, palette=None):
    df = df.copy()
    values = df[col].dropna()
    hist_fig = px.histogram(
        df,
        x=col,
        nbins=nbins,
        title=title or f"Distribution of {col}",
        color_discrete_sequence=palette or [colour[1]],
        opacity=0.7
    )
    kde = gaussian_kde(values)
    x_vals = np.linspace(values.min(), values.max(), 200)
    y_vals = kde(x_vals)

    hist_fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals * len(values) * (values.max() - values.min()) / nbins,
            mode='lines',
            name='Density',
            line=dict(color=palette[0] if palette else colour[0], width=2)
        )
    )

    hist_fig.update_layout(
        bargap=0.1,
        yaxis_title='Count',
        xaxis_title=col,
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.99)
    )
    hist_fig.show()

def plot_scatter_with_hue(df, x, y, hue, title=None):
    fig = px.scatter(
        df.copy(),
        x=x,
        y=y,
        color=hue,
        color_discrete_sequence=colour,
        title=title or f"{y} vs {x} colored by {hue}"
    )
    fig.show()

def plot_box_by_category(df, x, y, title=None, category_palette=colour):
    fig = px.box(
        df.copy(),
        x=x,
        y=y,
        color=x,
        color_discrete_sequence=category_palette,
        title=title or f"{y} by {x}"
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()

def plot_mean_by_group(df, group_col, value_col, title=None):
    means = df.groupby(group_col)[value_col].mean().reset_index().sort_values(value_col).copy()
    fig = px.bar(
        means,
        x=value_col,
        y=group_col,
        orientation='h',
        title=title or f"Average {value_col} by {group_col}",
        color_discrete_sequence=[colour[2]]
    )
    fig.show()

def plot_feature_importance(df, top_n=None, title="Feature Importance"):
    df = df.copy()
    if 'feature' not in df.columns:
        df.columns = ['feature', 'importance']
    df = df.sort_values(by='importance', ascending=True)
    if top_n:
        df = df.tail(top_n)

    fig = px.bar(
        df,
        x='importance',
        y='feature',
        color='feature',
        orientation='h',
        color_discrete_sequence=colour,
        title=title
    )
    fig.update_layout(showlegend=False)
    fig.show()



def plot_trend(df, x, y, title="Trend Line", xlabel=None, ylabel=None, color_index=1):
    fig = px.line(
        df,
        x=x,
        y=y,
        markers=True,
        title=title,
        color_discrete_sequence=[colour[color_index]]
    )
    fig.update_layout(
        xaxis_title=xlabel or x,
        yaxis_title=ylabel or y,
        xaxis_tickangle=-45
    )
    fig.show()


def get_category_color_map(df, column, palette=colour):
    categories = df[column].dropna().unique()
    categories_sorted = sorted(categories)
    repeat_factor = (len(categories_sorted) // len(palette)) + 1
    extended_palette = palette * repeat_factor
    return dict(zip(categories_sorted, extended_palette[:len(categories_sorted)]))


def plot_risk_distribution(df, column='risk_probability', nbins=30, title="Distribution of Risk Probability"):
    fig = px.histogram(df, x=column, nbins=nbins, title=title, color_discrete_sequence=[colour[2]])
    fig.update_layout(xaxis_title=column, yaxis_title="Count")
    fig.show()


def plot_top_risky_customers(df, score_col='risk_probability', id_col='customer_account', top_n=20, title="Top Risky Customers", color_map=None):
    top_risky = df[[id_col, score_col]].sort_values(by=score_col, ascending=False).head(top_n)

    palette = None
    if color_map:
        top_risky['color'] = top_risky[id_col].map(color_map)
        palette = top_risky['color'].tolist()

    fig = px.bar(
        top_risky,
        x=score_col,
        y=id_col,
        orientation='h',
        title=title,
        color=id_col,
        color_discrete_sequence=palette if palette else [colour[0]]
    )
    fig.update_layout(xaxis_title=score_col, yaxis_title=id_col)
    fig.show()



def plot_box_by_group(df, group_col, value_col, title=None, category_palette=None):
    fig = px.box(
        df,
        x=group_col,
        y=value_col,
        color=group_col,
        title=title or f"{value_col} by {group_col}",
        color_discrete_map=category_palette
    )
    fig.update_layout(xaxis_tickangle=45)
    fig.show()



def plot_avg_value_by_group(df, group_col, value_col, title=None, category_palette=None):
    group_avg = df.groupby(group_col)[value_col].mean().reset_index().sort_values(value_col, ascending=False)
    fig = px.bar(
        group_avg,
        x=group_col,
        y=value_col,
        title=title or f"Average {value_col} by {group_col}",
        color=group_col,
        color_discrete_map=category_palette
    )
    fig.update_layout(xaxis_tickangle=45, yaxis_title=value_col)
    fig.show()


def plot_model_leaderboard(df, palette=colour):
    """
    Visualise AutoGluon leaderboard (accuracy, training time, prediction time) using Plotly.
    
    Parameters:
        df (pd.DataFrame): Leaderboard DataFrame with 'model', 'score_val', 'fit_time', 'pred_time_val'
        palette (list): Custom hex color list
    """
    df = df.sort_values(by='score_val', ascending=False).copy()
    model_order = df['model'].tolist()

    # Color mapping per model (consistent across charts)
    color_map = {model: palette[i % len(palette)] for i, model in enumerate(model_order)}

    # Plot 1: Accuracy
    fig1 = px.bar(
        df,
        x='score_val',
        y='model',
        orientation='h',
        color='model',
        color_discrete_map=color_map,
        title="Model Accuracy (Validation Score)",
        text='score_val'
    )
    fig1.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig1.update_layout(xaxis_title='Validation Score', yaxis_title='Model', showlegend=False)     
    fig1.show()

    # Plot 2: Training Time
    fig2 = px.bar(
        df,
        x='fit_time',
        y='model',
        orientation='h',
        color='model',
        color_discrete_map=color_map,
        title="Model Training Time",
        text='fit_time'
    )
    fig2.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
    fig2.update_layout(xaxis_title='Training Time (s)', yaxis_title='Model', showlegend=False ,     width=600,
        height=400)
    fig2.show()

    # Plot 3: Prediction Time
    fig3 = px.bar(
        df,
        x='pred_time_val',
        y='model',
        orientation='h',
        color='model',
        color_discrete_map=color_map,
        title="Model Prediction Time",
        text='pred_time_val'
    )
    fig3.update_traces(texttemplate='%{text:.3f}s', textposition='outside')
    fig3.update_layout(xaxis_title='Prediction Time (s)', yaxis_title='Model', showlegend=False ,     width=600,
        height=400)
    fig3.show()

def plot_feature_importance(df, top_n=None, title="Feature Importance", palette=colour):
    """
    Plot feature importance using Plotly (horizontal bar chart).
    """

    # Only reset index if 'feature' is not already a column
    if 'feature' not in df.columns:
        if df.index.name is not None:  
            df = df.reset_index().rename(columns={df.index.name: 'feature'})
        else:  
            df = df.reset_index().rename(columns={'index': 'feature'})

    # Sort and trim
    df = df.sort_values(by='importance', ascending=False)
    if top_n:
        df = df.head(top_n)

    # Color mapping
    color_map = {feat: palette[i % len(palette)] for i, feat in enumerate(df['feature'])}

    fig = px.bar(
        df,
        x='importance',
        y='feature',
        orientation='h',
        color='feature',
        color_discrete_map=color_map,
        title=title
    )
    fig.update_layout(
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        showlegend=False,
        margin=dict(l=80, r=20, t=60, b=40),
    )
    fig.show()




def plot_risk_threshold_sensitivity(df, column='unpaid_ratio', thresholds=None, color="#d62828"):
    """
    Interactive plot showing how the number of risky customers changes
    as the threshold increases.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    data = {
        "Threshold": thresholds,
        "Risky Customers": [(df[column] > t).sum() for t in thresholds]
    }

    fig = px.line(
        pd.DataFrame(data),
        x="Threshold",
        y="Risky Customers",
        markers=True,
        title="Number of Customers Labeled Risky at Varying Thresholds",
        color_discrete_sequence=[color]
    )

    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=thresholds),
        yaxis_title="Count of Risky Customers",
        xaxis_title="Threshold on " + column,
        margin=dict(l=60, r=30, t=60, b=60),
             width=600,
        height=400
    )
    fig.show()




def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline', line=dict(dash='dash')))
    fig.update_layout(
        title=f"ROC Curve (AUC = {auc_score:.3f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=600,
        height=400
    )
    fig.show()


def plot_risk_distribution_by_class(df, score_col='risk_probability', target_col='target_risky'):
    fig = px.histogram(
        df, x=score_col, color=target_col,
        nbins=30,
        barmode='overlay',
        title="Risk Score Distribution by True Class",
        color_discrete_map={0: '#2a9d8f', 1: '#d62828'}
    )
    fig.update_layout(xaxis_title="Predicted Risk Score", yaxis_title="Count",      
                      width=600,
        height=400)
    fig.show()



def plot_precision_recall(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap_score = average_precision_score(y_true, y_pred_proba)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall'))
    fig.update_layout(
        title=f"Precision-Recall Curve (AP = {ap_score:.3f})",
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=600,
        height=400
    )
    fig.show()



def plot_bar_chart(
    df,
    x,
    y,
    title=None,
    palette=colour,
    top_n=None,
    xaxis_title=None,
    yaxis_title=None,
    orientation='h',
):
    """
    General-purpose horizontal bar chart with color palette support.

    Parameters:
        df (pd.DataFrame): Input data
        x (str): Column for bar length (e.g. numeric values)
        y (str): Column for categories (e.g. customer names)
        title (str): Plot title
        palette (list or dict): Colors
        top_n (int): Show only top N rows
        xaxis_title (str): Custom label for x-axis
        yaxis_title (str): Custom label for y-axis
    """
    df = df.copy()
    
    if top_n:
        df = df.sort_values(by=x, ascending=False).head(top_n)

    # Create color map if needed
    if isinstance(palette, list):
        color_map = {str(v): palette[i % len(palette)] for i, v in enumerate(df[y].astype(str))}
    elif isinstance(palette, dict):
        color_map = palette
    else:
        color_map = None

    fig = px.bar(
        df,
        x=x,
        y=y,
        orientation=orientation,
        color=y,
        color_discrete_map=color_map,
        title=title
    )

    fig.update_layout(
        xaxis_title=xaxis_title or x,
        yaxis_title=yaxis_title or y,
        showlegend=False,
        margin=dict(l=80, r=20, t=60, b=40)
    )

    fig.show()
