import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from utils.utils import init_connection, get_next_items, get_models
from gridfs import GridFS
from bson.objectid import ObjectId
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Benchmark",
    page_icon="👋",
    layout="wide",
    menu_items={
        'About': "# sdnn benchmark app!"
    }
)

st.title('SDNN Benchmark')

client = init_connection()

db = client.sdnn
benchmark_gridfs = GridFS(db, 'benchmark_files')

with st.sidebar:
#with st.container(border=True):
    #col1, col2, col3, col4, col5 = st.columns((1, 0.4, 0.4, 0.5, 0.5))

    selected_model_name = st.selectbox(
        'Model Name',
        get_next_items(client),
        placeholder="Select model...")

    selected_device = st.selectbox(
        'Device',
        get_next_items(client, selected_model_name),
        placeholder="Select device...")

    selected_sdk = st.multiselect(
        'SDK',
        get_next_items(client, selected_model_name, selected_device),
        get_next_items(client, selected_model_name, selected_device)[0],
        max_selections = 2,
        placeholder="Select sdk...",
        help="arm sdk version: r1p2, r1p4 ...",)

    selected_chip = st.multiselect(
        'Chip Type',
        get_next_items(client, selected_model_name, selected_device, selected_sdk),
        get_next_items(client, selected_model_name, selected_device, selected_sdk)[0],
        max_selections = 2,
        placeholder="Select chip...",
        help="x9 series chip: x9sp, x9cc ...",)

    selected_acc_level = st.multiselect(
        'Accuracy Level',
        get_next_items(client, selected_model_name, selected_device,
            selected_sdk, selected_chip),
        get_next_items(client, selected_model_name, selected_device,
            selected_sdk, selected_chip)[0],
        max_selections = 4,
        placeholder="Select accuracy level...",
        help="accuracy level: [0, 1, 2, 3]",)

    selected_date = st.multiselect(
        'Date Time',
        get_next_items(client, selected_model_name, selected_device,
            selected_sdk, selected_chip, selected_acc_level),
        placeholder="Select date...")

    profile_show = st.toggle('Show profile report?')
    aipu_graph_show = st.toggle('Show aipu graph?')

items = get_models(client, selected_model_name, selected_device, selected_sdk,
        selected_chip, selected_acc_level, selected_date)

lable_list = []
model_info_list = []
profile_html_list = []
graph_json_list = []
graph_png_list = []
fig_line = go.Figure()
fig_box = go.Figure()
nodes_dict = {}
nodes_dict["name"] = []
nodes_dict["device"] = []
nodes_dict["sdk"] = []
nodes_dict["chip"] = []
nodes_dict["acc_level"] = []
nodes_dict["date"] = []
nodes_dict["node"] = []
nodes_dict["raw_data"] = []

for item in items:
    data_list = []
    label_list = []
    if "deploy_json" in item and item["deploy_json"]:
        model_info_list.append(item["deploy_json"][0])
    if "profile" in item and item["profile"]:
        profile_html_list.append(item["profile"])
    if "graph_json" in item and item["graph_json"]:
        graph_json_list.append(item["graph_json"])
    if "graph_png" in item and item["graph_png"]:
        graph_png_list.append(item["graph_png"])
    for k, v in item["data"].items():
        nodes_dict["name"].append(item['name'])
        nodes_dict["device"].append(item['device'])
        nodes_dict["sdk"].append(item['sdk'])
        nodes_dict["chip"].append(item['chip'])
        nodes_dict["acc_level"].append(item['acc_level'])
        nodes_dict["date"].append(item['date'])
        nodes_dict["raw_data"].append(v)
        if "compass" in k:
            nodes_dict["node"].append("npu")
        elif "divide" in k:
            nodes_dict["node"].append("quant")
        elif "multiply" in k:
            nodes_dict["node"].append("dequant")
        elif "cast" in k:
            nodes_dict["node"].append("cast")
        else:
            nodes_dict["node"].append(k)

nodes_df = pd.DataFrame(nodes_dict)

with st.expander("Selected Model Table"):
    st.write(nodes_df)

with st.container(border=True):
    node_list = nodes_df["node"].unique()
    node_selected = st.multiselect(
            "Which node to plot?",
            node_list,
            ["npu"] if "npu" in node_list else node_list[0])
    if node_selected:
        raw_data_list = []
        plot_df = nodes_df[nodes_df["node"].isin(node_selected)]
        for index, row in plot_df.iterrows():
            lable_name = row["node"]
            if len(plot_df["sdk"].unique()) == 2:
                lable_name += "-" + row["sdk"]
            elif len(plot_df["chip"].unique()) == 2:
                lable_name += "-" + row["chip"]
            elif len(plot_df["acc_level"].unique()) > 1:
                lable_name += "-" + str(row["acc_level"])
            fig_line.add_trace(
                go.Scatter(
                    y=row["raw_data"],
                    name=lable_name,
                    mode='lines+markers',
                    line_shape='spline'))
            fig_box.add_trace(
                go.Box(
                    y=row["raw_data"],
                    name=lable_name,
                    boxpoints='all',
                    boxmean=True,
                    pointpos=-1.8))
            raw_data_list.append(row["raw_data"])
            lable_list.append(lable_name)
        fig_dist = ff.create_distplot(raw_data_list, lable_list, bin_size=.1)

        tab1, tab2 = st.tabs(["Line", "Histogram"])
        with tab1:
            line_col1, box_col2 = st.columns((2, 1))
            line_col1.plotly_chart(fig_line, use_container_width=True, theme="streamlit")
            box_col2.plotly_chart(fig_box, use_container_width=True, theme="streamlit")
        with tab2:
            st.plotly_chart(fig_dist, use_container_width=True, theme="streamlit")

with st.expander("Model Information"):
    st.write(model_info_list)

if profile_show and profile_html_list:
    with st.expander("Model Profile Report"):
        col_profile = st.columns(len(profile_html_list))
        for index, html in enumerate(profile_html_list):
            with col_profile[index]:
                st.header(lable_list[index], divider='rainbow')
                for f in html:
                    data = benchmark_gridfs.get(f)
                    components.html(data.read(), height = 600, scrolling=True)

if aipu_graph_show and graph_json_list:
    with st.expander("AIPU Graph Json"):
        col_graph_json = st.columns(len(graph_json_list))
        for index, json in enumerate(graph_json_list):
            with col_graph_json[index]:
                st.header(lable_list[index], divider='rainbow')
                st.write(json)

if aipu_graph_show and graph_png_list:
    with st.expander("AIPU Graph PNG"):
        col_graph_png = st.columns(len(graph_png_list))
        for index, png in enumerate(graph_png_list):
            with col_graph_png[index]:
                st.header(lable_list[index], divider='rainbow')
                data = benchmark_gridfs.get(png)
                st.image(data.read(), caption="aipu graph png")



                    #data = benchmark_gridfs.get(f)
                    #st.write(data.read(), unsafe_allow_html=True)
                #data = benchmark_gridfs.get(html)
        #data_np = np.asarray(data_list)
        #df = pd.DataFrame(data_np.T, columns=label_list)
    #with tab3:
    #    st.plotly_chart(fig_box, use_container_width=True, theme="streamlit")
        #st.plotly_chart(fig_test, use_container_width=True, theme="streamlit")
        #st.bar_chart(df)
        #x_index = np.linspace(0, len(data_list[0]), len(data_list[0]), dtype=int)
        #box_trace_list.append(go.Box(x=data_np[index], name=label, boxpoints='all'))
    #df_test = pd.DataFrame(test_dict).melt(var_name="device")
    #fig_test = px.line(df_test, y = "value", color='device', markers=True)
    #fig_test = px.histogram(df_test, x="value", color='device')
    #fig_test = px.histogram(df_test,  x="value", color='device', marginal="box",
    #                   hover_data=df_test.columns)
    #st.write(df_test)
    #fig_test = px.box(df_test, y="value", color="device", boxmode="overlay", points='all')
    #fig_test.update_traces(quartilemethod="linear", jitter=0, col=1)
    #fig_test.update_traces(quartilemethod="linear", jitter=0, row=2)
    #fig_test.update_traces(jitter=0, row=2)

#fig_box.update_traces()
#fig_box.update_traces(box_trace_list[0], jitter=0, col=1)
#fig_box.update_traces(box_trace_list[1], jitter=0, col=2)
#fig_box.update_traces(box_trace_list[2], jitter=0, col=3)
#fig_box.update_layout(
#    xaxis=dict(title='sdnn performance box figure', zeroline=False),
#    boxmode='group'
#)

