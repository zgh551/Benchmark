import streamlit as st

st.set_page_config(
    page_title="SDNN",
    page_icon="👋",
    layout="wide",
    menu_items={
        'About': "# sdnn app!"
    }
)
#st.set_page_config(layout="wide")
st.title('SDNN Home')
st.write("# Welcome to SDNN! 👋")
st.sidebar.success("Select a page above.")
st.markdown(
    """
    SDNN 是一个基于开源编译器框架TVM的端到端的AI编译器框架, Semidrive对TVM编译器框架做了适配，主要特性如下：

    - 支持操作系统： **Android** 、 **Linux** 和 **QNX** ;
    - 支持推理后端： **CPU** 、 **GPU** 、 **SlimAI** 和 **AIPU** ;
    - 支持开发及部署语言： **C++** 和 **Python** ;
    - 支持 ``异构`` 和 ``同构`` 模型部署模式;
    - 支持 ``多进程`` 和 ``多线程`` 应用的开发;
"""
)

