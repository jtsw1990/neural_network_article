'''Main script for streamlit application.'''

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_moons


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

def draw_neural_network(weights, biases):
    G = nx.DiGraph()

    # Add input nodes
    G.add_node("Input 1")
    G.add_node("Input 2")

    # Add hidden layer nodes
    for i in range(1, 4):
        G.add_node(f"Hidden {i}")

    # Add output layer nodes
    G.add_node("Output 1")
    G.add_node("Output 2")

    # Add edges (weights)
    for i in range(2):
        for j in range(3):
            G.add_edge(f"Input {i+1}", f"Hidden {j+1}", weight=weights[i*3 + j])

    for j in range(3):
        for k in range(2):
            G.add_edge(f"Hidden {j+1}", f"Output {k+1}", weight=weights[6 + j*2 + k])

    # Define positions with reduced height
    pos = {
        "Input 1": (-1, 0.2), "Input 2": (-1, -0.2),
        "Hidden 1": (0, 0.6), "Hidden 2": (0, 0), "Hidden 3": (0, -0.6),
        "Output 1": (1, 0.2), "Output 2": (1, -0.2),
    }

    plt.figure(figsize=(8, 5))  # Set figure size (width, height)

    # Draw the nodes
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000)

    # Draw the edges
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={k: f'{v:.2f}' for k, v in edge_labels.items()}, font_color='red')

    st.pyplot(plt)

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

    # The old contour function from matplotlib implementation
    #calculated_y = (
    #    (const1 / (1 + np.exp(-(w1 * df['feature_1'].values + w2 * df['feature_2'].values + b1)))) 
    #    + (const2 / (1 + np.exp(-(w3 * df['feature_1'].values + w4 * df['feature_2'].values + b2)))) 
    #    + (const3 / (1 + np.exp(-(w5 * df['feature_1'].values + w6 * df['feature_2'].values + b3)))) + b5 - b4
    #)

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
    # Calculation the accuracy
    #calculated_y = (
    #    (-1) * (
    #        (df['feature_1'].values * 
    #         (w1 * const1 + w3 * const2 + w5 * const3) +
    #         (b1 * const1 + b2 * const2 + b3 * const3 + b4 - b5)) /
    #        (w2 * const1 + w4 * const2 + w6 * const3)
    #    )
    #)
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

    return fig

st.title('Modelling with neural networks')
st.markdown('''

            A neural network being a universal approximator means that, given sufficient neurons and layers,
            it can approximate any continuous function to any desired degree of accuracy.
            This idea is formalized in the Universal Approximation Theorem, which states that even a simple
            feedforward network with a single hidden layer and non-linear activation functions can represent any continuous function on a closed interval.

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
    w1 = st.slider(label=r"$w_{1}$", min_value=-0.5, max_value=0.5, step=0.01, value=0.42)
    w5 = st.slider(label=r"$w_{5}$", min_value=-0.5, max_value=0.5, step=0.01, value=0.27)
    w9 = st.slider(label=r"$w_{9}$", min_value=-0.5, max_value=0.5, step=0.01, value=0.27)

with cw2:
    w2 = st.slider(label=r"$w_{2}$", min_value=-0.5, max_value=0.5, step=0.01, value=-0.07)
    w6 = st.slider(label=r"$w_{6}$", min_value=-0.5, max_value=0.5, step=0.01, value=0.55)
    w10 = st.slider(label=r"$w_{10}$", min_value=0.4, max_value=0.5, step=0.01, value=0.4)

with cw3:
    w3 = st.slider(label=r"$w_{3}$", min_value=-0.5, max_value=0.5, step=0.01, value=-0.17)
    w7 = st.slider(label=r"$w_{7}$", min_value=-0.5, max_value=0.5, step=0.01, value=-0.04)
    w11 = st.slider(label=r"$w_{11}$", min_value=0.4, max_value=0.5, step=0.01, value=0.43)

with cw4:
    w4 = st.slider(label=r"$w_{4}$", min_value=-0.5, max_value=0.5, step=0.01, value=-0.05)   
    w8 = st.slider(label=r"$w_{8}$", min_value=-1.0, max_value=0.5, step=0.01, value=-0.22)
    w12 = st.slider(label=r"$w_{12}$", min_value=-0.5, max_value=0.5, step=0.01, value=0.7)

with cb1:
    b1 = st.slider(label=r"$b_{1}$", min_value=-0.1, max_value=0.5, step=0.01, value=-0.1)
    b4 = st.slider(label=r"$b_{4}$", min_value=0.05, max_value=0.5, step=0.01, value=-0.6)
with cb2:
    b2 = st.slider(label=r"$b_{2}$", min_value=-0.1, max_value=0.5, step=0.01, value=0.0)
    b5 = st.slider(label=r"$b_{5}$", min_value=0.2, max_value=0.25, step=0.01, value=0.23)
with cb3:
    b3 = st.slider(label=r"$b_{3}$", min_value=-0.1, max_value=0.5, step=0.01, value=-0.6)


draw_neural_network(
    [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12], 
    [b1, b2, b3, b4, b5]
)


st.markdown('''
    The 2 classes (denoted by the colour of the marker) in the dataset is not linearly seperable.
    Play around with the sliders on the left to try and create a function that is able to separate the red and grey markers.
''')


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



