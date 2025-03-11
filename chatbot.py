import gradio as gr
import pandas as pd
import numpy as np
import asyncio
from workflow import main

current_df = pd.read_csv('menudata.csv')
# Global variables to store the latest data
current_urls = []

def chat_response(message, history):
    global current_df, current_urls
    
    # Get response from main
    ans, urls, item_ids = asyncio.run(main(message))
    
    # Update URLs and dataframe
    if urls:
        current_urls = urls
    else:
        current_urls = []
    if item_ids:
        current_df = current_df[current_df['item_id'].isin(item_ids)]
    else:
        current_df = current_df
    
    return ans

def show_data():
    return current_df, "\n".join(current_urls)

with gr.Blocks() as demo:
    # Chat interface in first row
    with gr.Row():
        chatbot = gr.ChatInterface(
            fn=chat_response,
            type='messages',
            title="MenuData.ai",
            description="Ask MenuData.ai any question",
            flagging_mode="manual",
            flagging_options=["Good", "Bad"]
        )
    
    # Button in second row
    with gr.Row():
        show_button = gr.Button("Show References and Data")
    
    # URLs textbox
    with gr.Row():
        url_box = gr.Textbox(
            label="Reference URLs",
            interactive=False,
            lines=4,
            value=""
        )
    
    # Dataframe in third row
    with gr.Row():
        df_display = gr.DataFrame(
            value=current_df,
            interactive=False
        )
    
    # Connect button to show both dataframe and URLs
    show_button.click(
        fn=show_data,
        outputs=[df_display, url_box]
    )

demo.launch()
