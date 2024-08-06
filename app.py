import gradio as gr
from SoccerFoulProject.model import Model

def generate_predictions(video,start,end,):
    model=Model(video_encoder_name='mc3_18',clip_aggregation='max',feat_dim=100)
    baseline_weights_path='runs/Experiment/weights/best.pth'
    model.load(baseline_weights_path)
    action_label,off_severity_video=model.predict(video,start=start,end=end)    
    return action_label,off_severity_video

def runapp():
    demo=gr.Interface(fn=generate_predictions,
                      inputs=['video',gr.Slider(minimum=0,maximum=5,label='Start (seconds)'),gr.Slider(minimum=0,maximum=5,label='End (seconds)')],
                      outputs=[gr.Text(label='Action Label'),gr.Text(label='Offence Severity Label')])
    demo.launch()
