from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import squarify  
import pandas as pd
import seaborn as sns
import squarify


sns.set(style="whitegrid")

colour = ['#2a9d8f', '#003049', '#d62828', '#f77f00', '#fcbf49', '#eae2b7']
def get_category_color_map(df, column, palette=colour):
    categories = df[column].dropna().unique()
    categories_sorted = sorted(categories)
    repeat_factor = (len(categories_sorted) // len(palette)) + 1
    extended_palette = palette * repeat_factor
    return dict(zip(categories_sorted, extended_palette[:len(categories_sorted)]))

def plot_missing_data(df):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull().T, cbar=False, cmap=["white", colour[2]])
    plt.title("Missing Values Heatmap")
    plt.xlabel("Rows")
    plt.ylabel("Columns")
    plt.tight_layout()
    plt.show()

def plot_categorical_counts(df, columns, top_n=None, orient='v', title="Count of", category_color_map=None):
    for col in columns:
        data = df[col].value_counts().nlargest(top_n) if top_n else df[col].value_counts()
        data = data.reset_index()
        data.columns = [col, 'count']
        plt.figure(figsize=(10, 5))
        if orient == 'h':
            sns.barplot(data=data, x='count', y=col, palette=category_color_map or colour)
        else:
            sns.barplot(data=data, x=col, y='count', palette=category_color_map or colour)
            plt.xticks(rotation=45)
        plt.title(f"{title} {col}")
        plt.tight_layout()
        plt.show()

def plot_treemap(
    df,
    category_col,
    value_col=None,
    title="Treemap",
    category_palette=None 
):
    """
    Plot a treemap using squarify. Supports:
    - counts of each category (if value_col is None), or
    - sum of a value column grouped by category.
    """

    # Prepare data
    if value_col:
        data = df.groupby(category_col)[value_col].sum().reset_index()
    else:
        data = df[category_col].value_counts().reset_index()
        data.columns = [category_col, 'count']
        value_col = 'count'

    # Compute percentages and labels
    total = data[value_col].sum()
    data['percentage'] = (data[value_col] / total * 100).round(1)
    data['label'] = data[category_col].astype(str) + '\n' + data['percentage'].astype(str) + '%'

    # Sort values descending (optional for nicer layout)
    data = data.sort_values(value_col, ascending=False)

    # Map colours
    if category_palette:
        colours = [category_palette.get(cat, "#cccccc") for cat in data[category_col]]
    else:
        colours = plt.cm.Set3.colors[:len(data)]

    # Plot treemap
    plt.figure(figsize=(10, 6))
    squarify.plot(
        sizes=data[value_col],
        label=data['label'],
        color=colours,
        alpha=0.9,
        pad=True,
        text_kwargs={'fontsize': 9}
    )
    plt.axis('off')
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_numerical_distribution(df, col, title=None, nbins=30, palette=None):
    values = df[col].dropna()
    plt.figure(figsize=(8, 4))
    sns.histplot(values, bins=nbins, kde=True, color=palette[0] if palette else colour[1])
    plt.title(title or f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_scatter_with_hue(df, x, y, hue, title=None):
    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette=colour)
    plt.title(title or f"{y} vs {x} coloured by {hue}")
    plt.tight_layout()
    plt.show()

def plot_box_by_category(df, x, y, title=None, category_palette=colour):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y, palette=category_palette)
    plt.title(title or f"{y} by {x}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_mean_by_group(df, group_col, value_col, title=None):
    means = df.groupby(group_col)[value_col].mean().reset_index().sort_values(value_col)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=means, x=value_col, y=group_col, palette=[colour[2]])
    plt.title(title or f"Average {value_col} by {group_col}")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(df, top_n=None, title="Feature Importance", palette=colour):
    """
    Plot feature importances using Seaborn.

    Parameters:
        df (pd.DataFrame): DataFrame with at least 'feature' and 'importance' columns,
                           or 2 columns where index is feature.
    """
    df = df.copy()


    # Sort and trim
    df = df.sort_values(by='importance', ascending=False)
    if top_n:
        df = df.head(top_n)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x='importance', y='index', palette=palette)
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


def plot_trend(df, x, y, title="Trend Line", xlabel=None, ylabel=None, color_index=1):
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=df, x=x, y=y, marker='o', color=colour[color_index])
    plt.title(title)
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_risk_distribution(df, column='risk_probability', nbins=30, title="Distribution of Risk Probability"):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column], bins=nbins, color=colour[2])
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_top_risky_customers(df, score_col='risk_probability', id_col='customer_account', top_n=20, title="Top Risky Customers", color_map=None):
    top_risky = df[[id_col, score_col]].sort_values(by=score_col, ascending=False).head(top_n)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=top_risky, x=score_col, y=id_col, palette=[colour[0]] if not color_map else top_risky[id_col].map(color_map))
    plt.title(title)
    plt.xlabel(score_col)
    plt.ylabel(id_col)
    plt.tight_layout()
    plt.show()

def plot_box_by_group(df, group_col, value_col, title=None, category_palette=None):
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x=group_col, y=value_col, palette=category_palette)
    plt.title(title or f"{value_col} by {group_col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_avg_value_by_group(df, group_col, value_col, title=None, category_palette=None):
    group_avg = df.groupby(group_col)[value_col].mean().reset_index().sort_values(value_col, ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=group_avg, x=group_col, y=value_col, palette=category_palette)
    plt.title(title or f"Average {value_col} by {group_col}")
    plt.xticks(rotation=45)
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.show()

def plot_model_leaderboard(df, palette=colour):
    df = df.sort_values(by='score_val', ascending=False)
    model_order = df['model'].tolist()
    color_map = {model: palette[i % len(palette)] for i, model in enumerate(model_order)}
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x='score_val', y='model', palette=color_map)
    plt.title("Model Accuracy (Validation Score)")
    plt.xlabel("Validation Score")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()

def plot_risk_threshold_sensitivity(df, column='unpaid_ratio', thresholds=None, color="#d62828"):
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    data = {
        "Threshold": thresholds,
        "Risky Customers": [(df[column] > t).sum() for t in thresholds]
    }
    plt.figure(figsize=(8, 4))
    sns.lineplot(x=data["Threshold"], y=data["Risky Customers"], marker='o', color=color)
    plt.title("Number of Customers Labelled Risky at Varying Thresholds")
    plt.xlabel("Threshold on " + column)
    plt.ylabel("Count of Risky Customers")
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(8, 4))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', color=colour[2])
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def plot_risk_distribution_by_class(df, score_col='risk_probability', target_col='target_risky'):
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=score_col, hue=target_col, bins=30, stat='count', palette={0: '#2a9d8f', 1: '#d62828'})
    plt.title("Risk Score Distribution by True Class")
    plt.xlabel("Predicted Risk Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_precision_recall(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap_score = average_precision_score(y_true, y_pred_proba)
    plt.figure(figsize=(8, 4))
    plt.plot(recall, precision, color=colour[0], label=f'AP = {ap_score:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

def plot_bar_chart(df, x, y, title=None, palette=colour, top_n=None, xaxis_title=None, yaxis_title=None, orientation='h'):
    df = df.copy()
    if top_n:
        df = df.sort_values(by=x, ascending=False).head(top_n)
    plt.figure(figsize=(8, 4))
    if orientation == 'h':
        sns.barplot(data=df, x=x, y=y, palette=palette)
        plt.xlabel(xaxis_title or x)
        plt.ylabel(yaxis_title or y)
    else:
        sns.barplot(data=df, x=y, y=x, palette=palette)
        plt.ylabel(xaxis_title or x)
        plt.xlabel(yaxis_title or y)
    plt.title(title)
    plt.tight_layout()
    plt.show()

