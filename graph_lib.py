import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
import lib

df = pd.read_csv(r'C:\Users\ayush\Downloads\openface\OpenFace_2.2.0_win_x64\processed\Lauren_Engagement.csv')
al = pd.read_csv(r'C:\Users\ayush\OneDrive\Documents\Ayushi_Lauren_CSV.csv')
feature = [0 for i in range(7)]
AU = {
    ' AU01_r': "Upper Brow Raiser",
    ' AU02_r': "Lower Brow Raiser",
    ' AU04_r': "Brow Lowerer",
    ' AU05_r': "Upper Lid Raiser",
    ' AU06_r': "Cheek Raiser",
    ' AU07_r':  "Lid Tightener",
    ' AU09_r': "Nose Wrinkler",
    ' AU10_r': "Upper Lip Raiser",
    ' AU12_r': "Lip Corner Puller",
    ' AU14_r': "Dimpler",
    ' AU15_r': "Lip Corner Depressor",
    ' AU17_r': "Chin Raiser",
    ' AU20_r': "Lip Stretcher",
    ' AU23_r': "Lip Tightener",
    ' AU25_r': "Lips Part",
    ' AU26_r': "Jaw Drop"
}
d_swap = {v: k for k, v in AU.items()}

facial_action = "Upper Brow Raiser"

def figure():
    # Create figure
    fig = go.Figure()

    # Add traces
    if feature[0] == 1:
        fig.add_trace(go.Scatter(
            x=df[' timestamp'],
            y=df[' x_{}'.format(lib.coor_dir["eyebrow"][0])],
            name='Eyebrow location'
        ))
    if feature[1] == 1:
        fig.add_trace(go.Scatter(
            x=df[' timestamp'],
            y=df[' x_{}'.format(lib.coor_dir["eye"][0])],
            name='Eye location'
        ))
    if feature[2] == 1:
        fig.add_trace(go.Scatter(
            x=df[' timestamp'],
            y=df[' x_{}'.format(lib.coor_dir["jaw"][0])],
            name='Jaw location'
        ))
    if feature[3] == 1:
        fig.add_trace(go.Scatter(
            x=df[' timestamp'],
            y=df[' x_{}'.format(lib.coor_dir["nose"][0])],
            name='Nose location'
        ))
    if feature[4] == 1:
        fig.add_trace(go.Scatter(
            x=df[' timestamp'],
            y=df[' x_{}'.format(lib.coor_dir["innerlip"][0])],
            name='Innerlip location'
        ))
    if feature[5] == 1:
        fig.add_trace(go.Scatter(
            x=df[' timestamp'],
            y=df[' x_{}'.format(lib.coor_dir["outerlip"][0])],
            name='Outerlip location'
        ))

    fig.update_layout(
        title="Location of Facial Features over Time",
        legend_title="Legend",
        font=dict(
            size=14,
            color="RebeccaPurple"
        )
    )
    if feature[6] == 1:
        eye_gaze = go.Figure()
        eye_gaze.add_trace(go.Scatter(
            x=df[' timestamp'],
            y=df[' gaze_angle_x'] ,
            name='Left-Right Eye Gaze location'
        ))
        eye_gaze.add_trace(go.Scatter(
            x=df[' timestamp'],
            y= df[' gaze_angle_y'],
            name='Up-Down Eye Gaze location'
        ))
        eye_gaze.update_layout(
            title="Eye Gaze over Time",
            legend_title="Legend",
            font=dict(
                size=14,
                color="RebeccaPurple"
            )
        )
        eye_gaze.update_traces(
            line={"width": 1},
            mode="lines",
            showlegend=True
        )
        eye_gaze.update_xaxes(title_text='Time in Seconds')
        eye_gaze.update_yaxes(title_text='Angle of Eye Gaze (In Radians)')
        eye_gaze.update_layout(
            shapes=[
                dict(
                    fillcolor="rgba(29, 22, 224, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=0,
                    x1=al['Column6'][0],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(224, 123, 22, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][0],
                    x1=al['Column6'][1],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(29, 22, 224, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][1],
                    x1=al['Column6'][2],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(224, 123, 22, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][2],
                    x1=al['Column6'][3],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(29, 22, 224, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][3],
                    x1=al['Column6'][4],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(224, 123, 22, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][4],
                    x1=al['Column6'][5],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(29, 22, 224, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][5],
                    x1=al['Column6'][6],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(224, 123, 22, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][6],
                    x1=al['Column6'][7],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(29, 22, 224, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][7],
                    x1=al['Column6'][8],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(224, 123, 22, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][8],
                    x1=al['Column6'][9],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(29, 22, 224, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][9],
                    x1=al['Column6'][10],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(224, 123, 22, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][10],
                    x1=al['Column6'][11],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(29, 22, 224, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][11],
                    x1=al['Column6'][12],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(224, 123, 22, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][12],
                    x1=al['Column6'][13],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(29, 22, 224, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][13],
                    x1=al['Column6'][14],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(224, 123, 22, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][14],
                    x1=al['Column6'][15],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(29, 22, 224, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][15],
                    x1=al['Column6'][16],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(224, 123, 22, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][16],
                    x1=al['Column6'][17],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(29, 22, 224, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][17],
                    x1=al['Column6'][18],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                ),
                dict(
                    fillcolor="rgba(224, 123, 22, 0.2)",
                    line={"width": 0},
                    type="rect",
                    x0=al['Column6'][18],
                    x1=al['Column6'][19],
                    xref="x",
                    y0=0,
                    y1=0.95,
                    yref="paper"
                )
            ]
        )

        eye_gaze.update_layout(
            xaxis=dict(
                autorange=True,
                range=[0, 389],
                rangeslider=dict(
                    autorange=True,
                    range=[0, 389]
                ),
            )
        )

        eye_gaze.update_layout(
            dragmode="zoom",
            hovermode="x",
            legend=dict(traceorder="reversed"),
            height=600,
            template="plotly_white",
            margin=dict(
                t=100,
                b=100
            ),
        )
        eye_gaze.show()

    # style all the traces
    fig.update_traces(
        line={"width": 1},
        mode="lines",
        showlegend=True
    )

    fig.update_xaxes(title_text='Time in Seconds')
    fig.update_yaxes(title_text='Location in pixels')

    # Add shapes
    fig.update_layout(
        shapes=[
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=0,
                x1=al['Column6'][0],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][0],
                x1=al['Column6'][1],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][1],
                x1=al['Column6'][2],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][2],
                x1=al['Column6'][3],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][3],
                x1=al['Column6'][4],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][4],
                x1=al['Column6'][5],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][5],
                x1=al['Column6'][6],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][6],
                x1=al['Column6'][7],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][7],
                x1=al['Column6'][8],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][8],
                x1=al['Column6'][9],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][9],
                x1=al['Column6'][10],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][10],
                x1=al['Column6'][11],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][11],
                x1=al['Column6'][12],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][12],
                x1=al['Column6'][13],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][13],
                x1=al['Column6'][14],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][14],
                x1=al['Column6'][15],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][15],
                x1=al['Column6'][16],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][16],
                x1=al['Column6'][17],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][17],
                x1=al['Column6'][18],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][18],
                x1=al['Column6'][19],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            )
        ]
    )

    fig.update_layout(
        xaxis=dict(
            autorange=True,
            range=[0, 389],
            rangeslider=dict(
                autorange=True,
                range=[0, 389]
            ),
        )
    )

    fig.update_layout(
        dragmode="zoom",
        hovermode="x",
        legend=dict(traceorder="reversed"),
        height=600,
        template="plotly_white",
        margin=dict(
            t=100,
            b=100
        ),
    )

    fig.show()

def AU_figure():
    # Create figure
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(
        x=df[' timestamp'],
        y=df[d_swap[facial_action]],
        name=facial_action
    ))

    fig.update_layout(
        title='{} Intensity over Time'.format(facial_action),
        legend_title="Legend",
        font=dict(
            size=14,
            color="RebeccaPurple"
        )
    )

    # style all the traces
    fig.update_traces(
        line={"width": 1},
        mode="lines",
        showlegend=True
    )

    fig.update_xaxes(title_text='Time in Seconds')
    fig.update_yaxes(title_text='{} Intensity'.format(facial_action))

    # Add shapes
    fig.update_layout(
        shapes=[
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=0,
                x1=al['Column6'][0],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][0],
                x1=al['Column6'][1],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][1],
                x1=al['Column6'][2],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][2],
                x1=al['Column6'][3],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][3],
                x1=al['Column6'][4],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][4],
                x1=al['Column6'][5],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][5],
                x1=al['Column6'][6],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][6],
                x1=al['Column6'][7],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][7],
                x1=al['Column6'][8],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][8],
                x1=al['Column6'][9],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][9],
                x1=al['Column6'][10],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][10],
                x1=al['Column6'][11],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][11],
                x1=al['Column6'][12],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][12],
                x1=al['Column6'][13],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][13],
                x1=al['Column6'][14],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][14],
                x1=al['Column6'][15],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][15],
                x1=al['Column6'][16],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][16],
                x1=al['Column6'][17],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(29, 22, 224, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][17],
                x1=al['Column6'][18],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(224, 123, 22, 0.2)",
                line={"width": 0},
                type="rect",
                x0=al['Column6'][18],
                x1=al['Column6'][19],
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            )
        ]
    )

    fig.update_layout(
        xaxis=dict(
            autorange=True,
            range=[0, 389],
            rangeslider=dict(
                autorange=True,
                range=[0, 389]
            ),
        )
    )

    fig.update_layout(
        dragmode="zoom",
        hovermode="x",
        legend=dict(traceorder="reversed"),
        height=600,
        template="plotly_white",
        margin=dict(
            t=100,
            b=100
        ),
    )

    fig.show()



def pred_figure(y_pred):
    output = go.Figure()

    time = [i / 25 for i in range(int(len(y_pred)))]
    time1 = np.array(time)
    y = []
    y1 = []
    for i in y_pred:
        if i == 1:
            y.append("Engaged")
        else:
            y.append("Disengaged")
    for i in lib.al_arr:
        if i == 1:
            y1.append("Engaged")
        else:
            y1.append("Disengaged")
    # Add traces
    output.add_trace(go.Scatter(
        x=time1,
        y=y,
        name="Model Prediction",
        line=dict(color='firebrick', width=4),
        yaxis="y2",
    ))
    output.add_trace(go.Scatter(
        x=time1,
        y=y1,
        name="Ground Truth",
        line=dict(color='royalblue', width=4,
                  dash='dot'),
        yaxis="y1",
    ))

    output.update_yaxes(title="Engagement")

    output.update_xaxes(title="Time in Seconds")
    output.update_yaxes(type='category')

    output.update_layout(
        autosize=False,
        width=900,
        height=400,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
    )
    output.update_layout(
        xaxis=dict(
            autorange=True,
            range=[0, 389],
            rangeslider=dict(
                autorange=True,
                range=[0, 389]
            ),
        )
    )

    output.update_layout(
        dragmode="zoom",
        hovermode="x",
        legend=dict(traceorder="reversed"),
        template="plotly_white",
        margin=dict(
            t=100,
            b=100
        ),
    )

    output.update_layout(
        xaxis=dict(
            autorange=True,
            rangeslider=dict(
                autorange=True
            ),
        ),
        yaxis=dict(
            anchor="x",
            autorange=True,
            domain=[0, 0.4],
            linecolor="#673ab7",
            mirror=True,
            showline=True,
            side="left",
            tickfont={"color": "#673ab7"},
            tickmode="auto",
            ticks="",
            titlefont={"color": "#673ab7"},
            type="category",
            zeroline=False
        ),
        yaxis2=dict(
            anchor="x",
            autorange=True,
            domain=[0.6, 1],
            linecolor="#E91E63",
            mirror=True,
            range=[29.3787777032, 100.621222297],
            showline=True,
            side="left",
            tickfont={"color": "#E91E63"},
            tickmode="auto",
            ticks="",
            titlefont={"color": "#E91E63"},
            type="category",
            zeroline=False
        ),
    )
    output.update_layout(
        title="Model Prediction and Ground Truth of Engagement",
        legend_title="Legend",
    )
    output.show()
