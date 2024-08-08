import streamlit as st
from streamlit import session_state as ss
from bson.objectid import ObjectId
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from utils.utils import *
import plotly.figure_factory as ff
import plotly.graph_objects as go

st.set_page_config(
    page_title="Performance Summarize",
    page_icon="👋",
    layout="wide",
    menu_items={
        'About': "# sdnn performance benchmark summarize app!"
    }
)

st.title('SDNN Performance Benchmark Summarize')

client = init_connection()

items = get_models_performance(client)

df_dict = {}
df_dict["show"] = []
df_dict["name"] = []
df_dict["category"] = []
df_dict["input_shape"] = []
df_dict["output_shape"] = []
df_dict["ptg"] = []
df_dict["sdk"] = []
df_dict["chip"] = []
df_dict["os"] = []
df_dict["device"] = []
df_dict["acc_level"] = []
df_dict["bit_quant"] = []
df_dict["winograd"] = []
df_dict["dtcm"] = []
## mean run time
df_dict["mean_quant"] = []
df_dict["mean_npu"] = []
df_dict["mean_dequant"] = []
df_dict["mean_rpc"] = []
## other info no display
df_dict["date"] = []
df_dict["rpc_time_data_id"] = []
df_dict["time_data_id"] = []
df_dict["deploy_json_id"] = []
df_dict["graph_json_id"] = []
df_dict["graph_png_id"] = []
df_dict["profile_html_id"] = []


for item in items:
    df_dict["show"].append(False)
    df_dict["name"].append(item['name'])
    df_dict["category"].append(item['category'])
    deploy_dict = get_json_file(client, item['deploy_json'])
    df_dict["input_shape"].append(get_shape(deploy_dict['inputs']))
    df_dict["output_shape"].append(get_shape(deploy_dict['outputs']))
    df_dict["ptg"].append(item['ptg'])
    df_dict["sdk"].append(item['sdk'])
    df_dict["chip"].append(item['chip'])
    df_dict["os"].append(item['os'])
    df_dict["device"].append(item['device'])
    df_dict["acc_level"].append(item['acc_level'])
    df_dict["bit_quant"].append(item['quant_bit'])
    df_dict["winograd"].append(item['winograd'])
    df_dict["dtcm"].append(item['dtcm'])
    ## calculate mean run time
    data_dict = {"mean_npu":None, "mean_quant":None, "mean_dequant":None, "mean_rpc":None}
    run_time_data = get_npz_file(client, item['time_data'])
    for k in run_time_data.files:
        if "compass" in k:
            data_dict["mean_npu"] = np.median(run_time_data[k])
        elif "divide" in k:
            data_dict["mean_quant"] = np.median(run_time_data[k])
        elif "multiply" in k:
            data_dict["mean_dequant"] = np.median(run_time_data[k])
        elif "cast" in k:
            data_dict["mean_dequant"] = np.median(run_time_data[k])
        else:
            #print(k)
            data_dict["mean_npu"] = np.median(run_time_data[k])
    if "rpmsg_time_data" in item.keys():
        if item['rpmsg_time_data']:
            rpc_run_time_data = get_npz_file(client, item['rpmsg_time_data'])
            for k in rpc_run_time_data.files:
                if "rpc-run" == k:
                    data_dict["mean_rpc"] =  np.median(rpc_run_time_data[k])
                else:
                    print("unkonw key: {}".format(k))
    for k,v in data_dict.items():
        df_dict[k].append(v)

    df_dict["date"].append(item['date'])
    if "rpmsg_time_data" in item.keys() and item['rpmsg_time_data']:
        df_dict["rpc_time_data_id"].append(item['rpmsg_time_data'].binary)
    else:
        df_dict["rpc_time_data_id"].append(None)
    df_dict["time_data_id"].append(item['time_data'].binary)
    df_dict["deploy_json_id"].append(item['deploy_json'].binary)
    df_dict["graph_json_id"].append(item['graph_json'].binary)
    df_dict["graph_png_id"].append(item['graph_png'].binary)
    df_dict["profile_html_id"].append([oid.binary for oid in item['profile_html']])

df = pd.DataFrame(df_dict)

with st.sidebar:
    #if not "filter_df" in ss:
    filter_df = filter_dataframe(df)

    st.header("Option Info", divider='rainbow')
    run_time_plot_show = st.toggle('Show Run Time Plot?')
    profile_show = st.toggle('Show profile html report?')
    aipu_graph_json_show = st.toggle('Show aipu graph json?')
    aipu_graph_png_show = st.toggle('Show aipu graph png?')

with st.container(border=True):
    st.data_editor(filter_df,
        column_order=["show", "name", "category", "input_shape", "output_shape",
            "ptg", "sdk", "chip", "os", "device", "acc_level", "bit_quant",
            "winograd", "dtcm", "mean_quant", "mean_npu", "mean_dequant",
            "mean_rpc"],
        disabled =["name", "category", "input_shape", "output_shape",
            "ptg","sdk", "chip", "os", "device", "acc_level", "bit_quant",
            "winograd", "dtcm", "mean_quant", "mean_npu", "mean_dequant",
            "mean_rpc"],
        column_config={
            "show": "Show",
            "name": "Model Name",
            "input_shape": "Input Shape",
            "output_shape": "Onput Shape",
            "chip": "Chip",
            "device": "Device",
            "sdk": "SDK",
            "os": "OS",
            "ptg": "PTG",
            "acc_level": "Level",
            "mean_npu": "NPU",
            "mean_quant": "Quant",
            "mean_dequant": "Dequant",
            "mean_rpc": "RPC",
            "date": "Date",
        },
        #height = 500,
        hide_index = True,
        key="performance_df",
        use_container_width = True,)

#st.write(ss.performance_df)

edt_rows = ss.performance_df["edited_rows"]

is_show_df_index_list = []
if not len(edt_rows) == 0:
    for k,v in edt_rows.items():
        is_show = bool(v["show"])
        if is_show:
            is_show_df_index_list.append(k)

if len(is_show_df_index_list) > 0:
    ## record selected performance info
    performance_info_dict = {}
    for index in is_show_df_index_list:
        ## multi data compare add some info
        aux_lable_name = ""
        if len(filter_df["sdk"].unique()) >= 2:
            aux_lable_name += "-" + filter_df.iloc[index]["sdk"]
        if len(filter_df["ptg"].unique()) >= 2:
            aux_lable_name += "-" + filter_df.iloc[index]["ptg"]
        if len(filter_df["chip"].unique()) >= 2:
            aux_lable_name += "-" + filter_df.iloc[index]["chip"]
        if len(filter_df["os"].unique()) >= 2:
            aux_lable_name += "-" + filter_df.iloc[index]["os"]
        if len(filter_df["device"].unique()) >= 2:
            aux_lable_name += "-" + filter_df.iloc[index]["device"]
        if len(filter_df["acc_level"].unique()) >= 2:
            aux_lable_name += "-" + str(filter_df.iloc[index]["acc_level"])
        if len(filter_df["bit_quant"].unique()) >= 2:
            aux_lable_name += "-" + filter_df.iloc[index]["bit_quant"]
        model_name = filter_df.iloc[index]["name"] + aux_lable_name
        performance_info_dict[model_name] = {
            "rpc_time_data": filter_df.iloc[index]["rpc_time_data_id"],
            "time_data": filter_df.iloc[index]["time_data_id"],
            "deploy_json": filter_df.iloc[index]["deploy_json_id"],
            "graph_json": filter_df.iloc[index]["graph_json_id"],
            "graph_png": filter_df.iloc[index]["graph_png_id"],
            "profile_html": filter_df.iloc[index]["profile_html_id"],
        }
    ## run time plot
    if run_time_plot_show and performance_info_dict:
        fig_line = go.Figure()
        fig_box = go.Figure()
        node_selected = st.radio("Which node to plot?",
                [":rainbow[NPU]", ":rainbow[RPC]", 'Dequant', 'Quant'], 0,
                horizontal=True)
        ## run time plot figure
        for key, val in performance_info_dict.items():
            if val["rpc_time_data"]:
                rpc_run_time_data = get_npz_file(client, ObjectId(val["rpc_time_data"]))
            run_time_data = get_npz_file(client, ObjectId(val["time_data"]))
            lable_name = None
            selected_node_time = None
            if node_selected == ":rainbow[RPC]" and rpc_run_time_data:
                selected_node_time = rpc_run_time_data["rpc-run"]
                lable_name = "rpc"
            else:
                for node_name in run_time_data.files:
                    ## setup run time node name
                    if node_selected == ":rainbow[NPU]" and "compass" in node_name:
                        selected_node_time = run_time_data[node_name]
                        lable_name = "npu"
                        break
                    elif node_selected == "Dequant" and ("multiply" in node_name or "cast" in node_name):
                        selected_node_time = run_time_data[node_name]
                        lable_name = "dequant"
                        break
                    elif node_selected == "Quant" and "divide" in node_name:
                        selected_node_time = run_time_data[node_name]
                        lable_name = "quant"
                        break
            if lable_name:
                ## update plot data
                fig_line.add_trace(
                    go.Scatter(
                        y=selected_node_time,
                        name=lable_name + key,
                        mode='lines+markers',
                        line_shape='spline'))
                fig_box.add_trace(
                    go.Box(
                        y=selected_node_time,
                        name=lable_name + key,
                        boxpoints='all',
                        boxmean=True,
                        pointpos=-1.8))
        ## plotly chart
        with st.container(border=True):
            if lable_name:
                line_col1, box_col2 = st.columns((62, 38))
                line_col1.plotly_chart(fig_line, use_container_width=True, theme="streamlit")
                box_col2.plotly_chart(fig_box, use_container_width=True, theme="streamlit")
    ## selected whether show profile html report
    if profile_show and performance_info_dict:
        with st.container(border=True):
            col_profile = st.columns(len(performance_info_dict))
            for index, key in enumerate(performance_info_dict.keys()):
                with col_profile[index]:
                    st.header(key, divider='rainbow')
                    for f in performance_info_dict[key]["profile_html"]:
                        data = get_models_performance_files(client, ObjectId(f))
                        components.html(data.read(), height = 600, scrolling=True)
    ## selected whether show graph json
    if aipu_graph_json_show and performance_info_dict:
        with st.container(border=True):
            col_graph_json = st.columns(len(performance_info_dict))
            for index, json in enumerate(performance_info_dict.keys()):
                with col_graph_json[index]:
                    st.header(json, divider='rainbow')
                    st.json(get_json_file(client,
                        ObjectId(performance_info_dict[json]["graph_json"])))
    ## selected whether show graph png
    if aipu_graph_png_show and performance_info_dict:
        with st.container(border=True):
            col_graph_png = st.columns(len(performance_info_dict))
            for index, png in enumerate(performance_info_dict.keys()):
                with col_graph_png[index]:
                    st.header(png, divider='rainbow')
                    data = get_models_performance_files(client,
                            ObjectId(performance_info_dict[png]["graph_png"]))
                    st.image(data.read(), caption="aipu graph png")


