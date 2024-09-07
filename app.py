'''Main script for streamlit application.'''

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_moons


st.title('Neural Networks')
st.markdown('### Universal Approximation Theorem Visualized')

st.markdown('''
            The 2 classes (denoted by colour) in the dataset is not linearly seperable.

            A single-layer network using the identity activation with 1 node
            and 1 bias would resemble a simple linear regression model.
            These results are closed under addtion, which means that any linear
            combination of linear equations would still be linear. We can introduce
            non-linearity by changing the activation function in each node. This
            in a $2 x 3 x 2$ network using both activations.

            ''')

# Read in data
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
df = pd.DataFrame(
    dict(
        feature_1=X[:, 1] * 5,
        feature_2=X[:, 0] * 5,
        response=y
    )
)

# Sliders

st.sidebar.markdown('### Select Weights')
cw1, cw2, cw3, cw4 = st.sidebar.columns(4)

st.sidebar.markdown('### Select Biases')
cb1, cb2, cb3 = st.sidebar.columns(3)

with cw1:
    w1 = st.slider(label=r"$w_{1}$", min_value=-0.5, max_value=0.5, step=0.01, value=0.0)
    w5 = st.slider(label=r"$w_{5}$", min_value=-0.5, max_value=0.5, step=0.01, value=0.1)
    w9 = st.slider(label=r"$w_{9}$", min_value=-0.5, max_value=0.5, step=0.01, value=0.3)

with cw2:
    w2 = st.slider(label=r"$w_{2}$", min_value=-0.5, max_value=0.5, step=0.01, value=0.0)
    w6 = st.slider(label=r"$w_{6}$", min_value=-0.5, max_value=0.5, step=0.01, value=1.3)
    w10 = st.slider(label=r"$w_{10}$", min_value=0.4, max_value=0.5, step=0.01, value=0.4)

with cw3:
    w3 = st.slider(label=r"$w_{3}$", min_value=-0.5, max_value=0.5, step=0.01, value=0.0)
    w7 = st.slider(label=r"$w_{7}$", min_value=-0.5, max_value=0.5, step=0.01, value=0.6)
    w11 = st.slider(label=r"$w_{11}$", min_value=0.4, max_value=0.5, step=0.01, value=0.43)

with cw4:
    w4 = st.slider(label=r"$w_{4}$", min_value=-0.5, max_value=0.5, step=0.01, value=0.1)   
    w8 = st.slider(label=r"$w_{8}$", min_value=-1.0, max_value=0.5, step=0.01, value=0.0)
    w12 = st.slider(label=r"$w_{12}$", min_value=-0.5, max_value=0.5, step=0.01, value=0.7)

with cb1:
    b1 = st.slider(label=r"$b_{1}$", min_value=-0.1, max_value=0.5, step=0.01, value=-0.1)
    b4 = st.slider(label=r"$b_{4}$", min_value=0.05, max_value=0.5, step=0.01, value=-0.6)
with cb2:
    b2 = st.slider(label=r"$b_{2}$", min_value=-0.1, max_value=0.5, step=0.01, value=0.0)
    b5 = st.slider(label=r"$b_{5}$", min_value=0.2, max_value=0.25, step=0.01, value=0.23)
with cb3:
    b3 = st.slider(label=r"$b_{3}$", min_value=-0.1, max_value=0.5, step=0.01, value=-0.6)

# Plot data
def plot_data(df):

    df['response'] = df['response'].astype(str)
    fig = px.scatter(
        df,
        x="feature_1",
        y="feature_2",
        color="response",
        labels={'response': 'Class'},
        color_discrete_sequence=['grey', 'red']
    )
    fig.update_layout(
        width=350,
        height=350,
        xaxis_range=[-7.5, 12],
        yaxis_range=[-7.5, 12],
        xaxis=dict(showgrid=False, zeroline=False, visible=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, showticklabels=False),
        showlegend=False
    )
    fig.update_coloraxes(showscale=False)
    
    return fig

# Function to classify points based on the decision boundary
def classify_points_0(Y, F):
    classified_as_0 = Y < F
    return classified_as_0

# Function to calculate percentage of correctly classified points
def calculate_accuracy(df, F):
    classified_as_0 = classify_points_0(df['feature_2'].values, F)
    actual_class_0 = df['response'].astype(int) == 0

    correct_classifications = np.sum(classified_as_0 == actual_class_0)
    total_points = len(df)
    accuracy = (correct_classifications / total_points) * 100
    return accuracy


def plot_sigmoid(
    w1, w2, w3, w4,
    w5, w6, w7, w8,
    w9, w10, w11, w12,
    b1, b2, b3, b4, b5,
):
    const1 = w7 - w10
    const2 = w8 - w11
    const3 = w9 - w12
    xlist = np.linspace(-20, 20, 100)
    ylist = np.linspace(-20, 20, 100)
    X, Y = np.meshgrid(xlist, ylist)

    fig = plot_data(df)

    F = (
        (const1 / (1 + np.exp(-(w1 * X + w2 * Y + b1)))) 
        + (const2 / (1 + np.exp(-(w3 * X + w4 * Y + b2)))) 
        + (const3 / (1 + np.exp(-(w5 * X + w6 * Y + b3)))) + b5 - b4
    )

    calculated_y = (
        (const1 / (1 + np.exp(-(w1 * df['feature_1'].values + w2 * df['feature_2'].values + b1)))) 
        + (const2 / (1 + np.exp(-(w3 * df['feature_1'].values + w4 * df['feature_2'].values + b2)))) 
        + (const3 / (1 + np.exp(-(w5 * df['feature_1'].values + w6 * df['feature_2'].values + b3)))) + b5 - b4
    )

    # Fill the region below the decision boundary line
    fig.add_trace(go.Contour(
        z=np.where(F <= 0, 1, np.nan),  # Mask to fill below the boundary
        x=xlist,
        y=ylist,
        contours=dict(
            start=0,
            end=1,
            coloring='fill',
            showlabels=False
        ),
        showscale=False,
        showlegend=False,
        colorscale=[
            [0, 'rgba(255, 0, 0, 0.3)'],
            [1, 'rgba(255, 0, 0, 0.3)']
        ],
        opacity=0.3
    ))

    # Calculate accuracy
    accuracy = calculate_accuracy(df, calculated_y)
    st.write(f"**Classification Accuracy**: {accuracy:.2f}%")

    return fig

def plot_identity(
    w1, w2, w3, w4,
    w5, w6, w7, w8,
    w9, w10, w11, w12,
    b1, b2, b3, b4, b5,
):
    const1 = w7 - w10
    const2 = w8 - w11
    const3 = w9 - w12

    fig = plot_data(df)

    db_range = np.arange(-20, 20, 0.4)

    boundary = (
        (-1) * (
            (db_range * 
             (w1 * const1 + w3 * const2 + w5 * const3) +
             (b1 * const1 + b2 * const2 + b3 * const3 + b4 - b5)) /
            (w2 * const1 + w4 * const2 + w6 * const3)
        )
    )
    calculated_y = (
        (-1) * (
            (df['feature_1'].values * 
             (w1 * const1 + w3 * const2 + w5 * const3) +
             (b1 * const1 + b2 * const2 + b3 * const3 + b4 - b5)) /
            (w2 * const1 + w4 * const2 + w6 * const3)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=db_range,
            y=boundary,
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0)'),
            showlegend=False
            )
        )
    fig.add_trace(
        go.Scatter(
            x=db_range,
            y=np.full(len(db_range), -50), 
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.3)',
            mode='none',
            showlegend=False
            )
        )
    
    # Calculate accuracy
    accuracy = calculate_accuracy(df, calculated_y)
    st.write(f"**Classification Accuracy**: {accuracy:.2f}%")

    return fig

p1, p2 = st.columns(2)

with p1:
    st.markdown('**Identity Activation**: $f(x) = x$')
    st.plotly_chart(plot_identity(
        w1, w2, w3, w4,
        w5, w6, w7, w8,
        w9, w10, w11, w12,
        b1, b2, b3, b4, b5
    ))

with p2:
    st.markdown('**Sigmoid Activation**: $f(x) =  \\frac{1}{(1 + e^{-x})}$')
    st.plotly_chart(plot_sigmoid(
        w1, w2, w3, w4,
        w5, w6, w7, w8,
        w9, w10, w11, w12,
        b1, b2, b3, b4, b5
    ))

st.write(df.loc[df['response'] != '1', :].shape)
