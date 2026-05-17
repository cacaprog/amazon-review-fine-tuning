"""Gradio demo: paste a furniture review → dissatisfaction risk score + IG heatmap."""

import argparse
import os

import gradio as gr

from dissatisfaction_classifier.inference.predictor import DissatisfactionPredictor

_CHECKPOINT = os.getenv("CHECKPOINT_DIR", "outputs/checkpoints/best")
_CONFIG = os.getenv("MODEL_CONFIG", "configs/model_config.yaml")
_CALIBRATION = os.getenv("CALIBRATION_PATH", "outputs/calibration/isotonic.pkl")

predictor = DissatisfactionPredictor(
    checkpoint_dir=_CHECKPOINT,
    config_path=_CONFIG,
    calibration_path=_CALIBRATION,
)


def _predict(text: str) -> tuple[float, str]:
    if not text or not text.strip():
        return 0.0, "<p>Please enter a review.</p>"
    result = predictor.predict(text)
    risk = round(result["risk_score"], 4)
    html = result["explanation"]["html"]
    return risk, html


with gr.Blocks(title="Dissatisfaction Early Warning") as demo:
    gr.Markdown(
        "## Critical Dissatisfaction Early Warning\n"
        "Paste an Amazon furniture review. The model returns a dissatisfaction risk score "
        "(probability of 1–2★ sentiment) and highlights the tokens that drove the prediction."
    )
    with gr.Row():
        text_input = gr.Textbox(
            label="Review text",
            placeholder="Paste a furniture review here…",
            lines=5,
        )
    with gr.Row():
        score_output = gr.Number(label="Dissatisfaction Risk Score [0–1]")
        heatmap_output = gr.HTML(label="Token Attribution (red = dissatisfaction, blue = satisfaction)")
    submit_btn = gr.Button("Analyse", variant="primary")
    submit_btn.click(fn=_predict, inputs=text_input, outputs=[score_output, heatmap_output])

    gr.Examples(
        examples=[
            ["Fell apart after two days. Screws stripped immediately and customer service was useless."],
            ["Beautiful solid wood table. Assembly was straightforward and it looks exactly like the photos."],
        ],
        inputs=text_input,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo.launch(server_port=args.port, share=args.share)
